//! Decodes Cursor checkpoint message data into canonical Messages.
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::Value;

use crate::{
    model::{
        image_from_encoded, is_image_mime, CanonicalMessage, ContentPart, MessageContent, Origin,
        RecoveredToolRound, Role, ToolCall, ToolCallContent, ToolImageReference, ToolResultContent,
        ToolRoundAssistant, ToolRoundId,
    },
    store::BlobId,
    Error, Result,
};

use super::REPLAY_ENVELOPE_PREFIX;

pub fn decode(data: &[u8], internal_id: String) -> Result<CanonicalMessage> {
    let value: Value = serde_json::from_slice(data)?;
    let role = match required_string(&value, "role")? {
        "system" => Role::System,
        "user" => Role::User,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        role => {
            return Err(Error::Protocol(format!(
                "unknown Cursor message role: {role}"
            )))
        }
    };
    let wire_id = value
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let is_request_context = role == Role::User && wire_id.starts_with("request-context:");
    let is_prompt_context =
        is_request_context || role == Role::User && wire_id.starts_with("selected-context:");
    let origin = match role {
        Role::System => Origin::Prompt,
        Role::Assistant => Origin::Assistant,
        Role::Tool => Origin::Tool,
        Role::User if wire_id.starts_with("runtime:") => Origin::Runtime,
        Role::User if is_prompt_context => Origin::Prompt,
        Role::User => Origin::User,
    };
    let runtime_event_id = wire_id.strip_prefix("runtime:").map(str::to_string);
    let content = match role {
        Role::Assistant => decode_assistant(&value, &internal_id)?,
        Role::Tool => MessageContent::ToolResult(decode_tool_result(&value)?),
        _ => decode_text(&value)?,
    };
    let message_id = if runtime_event_id.is_some() || is_request_context {
        wire_id
    } else {
        internal_id
    };
    Ok(CanonicalMessage {
        message_id,
        role,
        origin,
        content,
        runtime_event_id,
    })
}

pub fn decode_pending(value: &str) -> Result<RecoveredToolRound> {
    let wire: Value = serde_json::from_str(value)?;
    let started_at_ms = wire
        .pointer("/providerOptions/cursor/pendingToolCallStartedAtMs")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            Error::Protocol("Cursor pending assistant is missing pendingToolCallStartedAtMs".into())
        })?;
    let internal_id = format!(
        "cursor-pending:{}",
        BlobId::digest(value.as_bytes()).to_base64()
    );
    let message = decode(value.as_bytes(), internal_id.clone())?;
    let MessageContent::Assistant {
        text,
        thinking,
        tool_round_id: _,
        replay_state,
        tool_calls,
    } = message.content
    else {
        return Err(Error::Protocol(
            "Cursor pending message is not an assistant message".into(),
        ));
    };
    if tool_calls.is_empty() {
        return Err(Error::Protocol(
            "Cursor resume contains a pending assistant without tool calls".into(),
        ));
    }
    let model_call_id = wire
        .pointer("/providerOptions/cursor/modelProviderMessageId")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or(&internal_id)
        .to_string();
    let calls = tool_calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| {
            let argument_error = wire
                .pointer("/providerOptions/cursor/pendingToolExecutionContracts")
                .and_then(|contracts| contracts.get(&call.call_id))
                .and_then(|contract| contract.get("argumentError"))
                .and_then(Value::as_str)
                .map(str::to_string);
            Ok(ToolCall {
                index,
                call_id: call.call_id,
                model_call_id: model_call_id.clone(),
                name: call.name,
                arguments_text: serde_json::to_string(&call.arguments)?,
                arguments: call.arguments,
                argument_error,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecoveredToolRound {
        assistant: ToolRoundAssistant {
            text,
            thinking,
            model_call_id,
            replay_state,
        },
        calls,
        started_at_ms,
    })
}

fn decode_text(value: &Value) -> Result<MessageContent> {
    let content = value.get("content").unwrap_or(&Value::Null);
    if let Some(text) = content.as_str() {
        return Ok(MessageContent::Parts {
            parts: vec![ContentPart::Text { text: text.into() }],
        });
    }
    let parts = content
        .as_array()
        .ok_or_else(|| Error::Protocol("Cursor message content is not an array".into()))?
        .iter()
        .map(|part| match part.get("type").and_then(Value::as_str) {
            Some("text") => Ok(ContentPart::Text {
                text: required_string(part, "text")?.into(),
            }),
            Some("image" | "image_url" | "input_image" | "file") => image_part(part),
            Some(kind) => Err(Error::Protocol(format!(
                "unsupported Cursor message content part: {kind}"
            ))),
            None => Err(Error::Protocol(
                "Cursor message content part is missing type".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(MessageContent::Parts { parts })
}

fn decode_assistant(value: &Value, internal_id: &str) -> Result<MessageContent> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut calls = Vec::new();
    let mut replay_state = None;
    for part in value
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default())
            }
            Some("reasoning") => {
                thinking.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
                if let Some(signature) = part.get("signature").and_then(Value::as_str) {
                    if replay_state.is_some() {
                        return Err(Error::Protocol(
                            "Cursor assistant has multiple reasoning signatures".into(),
                        ));
                    }
                    replay_state = Some(decode_replay_state(signature)?);
                }
            }
            Some("tool-call") => calls.push(ToolCallContent {
                index: calls.len(),
                call_id: required_string(part, "toolCallId")?.into(),
                name: required_string(part, "toolName")?.into(),
                arguments: part.get("args").cloned().unwrap_or(Value::Null),
            }),
            _ => {}
        }
    }
    Ok(MessageContent::Assistant {
        text,
        thinking,
        tool_round_id: (!calls.is_empty())
            .then(|| ToolRoundId::new(format!("{internal_id}:tool-round"))),
        replay_state,
        tool_calls: calls,
    })
}

fn decode_replay_state(signature: &str) -> Result<crate::model::ProviderReplayState> {
    let Some(encoded) = signature.strip_prefix(REPLAY_ENVELOPE_PREFIX) else {
        return Ok(crate::model::ProviderReplayState {
            provider_kind: "cursor_opaque".into(),
            value: Value::String(signature.into()),
        });
    };
    let bytes = STANDARD.decode(encoded).map_err(|error| {
        Error::Protocol(format!(
            "invalid Cursor BYOK replay envelope base64: {error}"
        ))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::Protocol(format!("invalid Cursor BYOK replay envelope: {error}")))
}

fn decode_tool_result(value: &Value) -> Result<ToolResultContent> {
    let part = value
        .get("content")
        .and_then(Value::as_array)
        .and_then(|parts| parts.first())
        .ok_or_else(|| Error::Protocol("Cursor tool message has no result part".into()))?;
    let content = part
        .get("result")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let (image, mut provider_parts) = experimental_images(part)?;
    if !provider_parts.is_empty() {
        provider_parts.insert(
            0,
            ContentPart::Text {
                text: content.clone(),
            },
        );
    }
    Ok(ToolResultContent {
        call_id: required_string(part, "toolCallId")?.into(),
        name: required_string(part, "toolName")?.into(),
        content,
        is_error: part
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        image,
        provider_parts,
    })
}

fn experimental_images(part: &Value) -> Result<(Option<ToolImageReference>, Vec<ContentPart>)> {
    let Some(items) = part.get("experimental_content").and_then(Value::as_array) else {
        return Ok((None, Vec::new()));
    };
    let mut image = None;
    let mut parts = Vec::new();
    for item in items {
        if let Some("image" | "image_url" | "input_image" | "file") =
            item.get("type").and_then(Value::as_str)
        {
            if let Some(reference) = blob_image(item)? {
                image.get_or_insert(reference);
            }
            if image_payload(item).is_some() {
                parts.push(image_part(item)?);
            }
        }
    }
    Ok((image, parts))
}

fn blob_image(part: &Value) -> Result<Option<ToolImageReference>> {
    let Some(blob_id) = part.get("blobId").and_then(Value::as_str) else {
        return Ok(None);
    };
    if blob_id.is_empty() {
        return Err(Error::Protocol(
            "Cursor tool image is missing blobId".into(),
        ));
    }
    let mime_type = part
        .get("mimeType")
        .or_else(|| part.get("mediaType"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !is_image_mime(mime_type) {
        return Err(Error::Protocol(format!(
            "Cursor tool image has invalid MIME type: {mime_type}"
        )));
    }
    Ok(Some(ToolImageReference {
        blob_id: blob_id.into(),
        mime_type: mime_type.into(),
        path: part
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .into(),
    }))
}

fn image_part(part: &Value) -> Result<ContentPart> {
    let kind = part.get("type").and_then(Value::as_str).unwrap_or("image");
    let hint = part
        .get("mimeType")
        .or_else(|| part.get("mediaType"))
        .or_else(|| part.get("media_type"))
        .or_else(|| part.pointer("/source/media_type"))
        .and_then(Value::as_str);
    if kind == "file" && !hint.is_some_and(is_image_mime) {
        return Err(Error::Protocol("Cursor file part is not an image".into()));
    }
    let payload = image_payload(part)
        .ok_or_else(|| Error::Protocol("Cursor image part is missing image data".into()))?;
    let (mime_type, data) = image_from_encoded(payload, hint)?;
    Ok(ContentPart::Image { mime_type, data })
}

fn image_payload(part: &Value) -> Option<&str> {
    ["image", "data", "file_data", "url"]
        .into_iter()
        .find_map(|name| string_or_url(part.get(name)))
        .or_else(|| part.pointer("/image_url/url").and_then(Value::as_str))
        .or_else(|| string_or_url(part.get("image_url")))
        .or_else(|| part.pointer("/source/data").and_then(Value::as_str))
}

fn string_or_url(value: Option<&Value>) -> Option<&str> {
    let value = value?;
    value
        .as_str()
        .or_else(|| value.get("url").and_then(Value::as_str))
}

fn required_string<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Protocol(format!("Cursor message is missing {name}")))
}
