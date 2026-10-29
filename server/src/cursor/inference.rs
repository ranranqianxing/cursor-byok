//! Serves Cursor 3.27 local agent runs that speak `RunInference` instead of `AgentClientMessage`.
//!
//! The desktop opens `InferenceService/RunInferenceSSE` and appends
//! `RunInferenceClientMessage` frames on `BidiAppend`. A run request names the
//! BYOK model. After `run_ready`, `invoke_model` carries the conversation,
//! including image parts, and this module streams the provider reply back as
//! inference text parts.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use futures_util::StreamExt;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use crate::{
    cursor::{
        protocol::{
            connect,
            inference::{
                self, inference_content_part, inference_core_message, inference_stream_response,
                run_inference_client_message, run_inference_server_message, InferenceContentPart,
                InferenceCoreMessage, InferenceModelParameterValue, InferenceStreamResponse,
                InferenceTextStreamPart, InferenceThinkingStreamPart, RunInferenceAgentModeConfig,
                RunInferenceInvocationEnd, RunInferenceInvocationError,
                RunInferenceInvocationResponse, RunInferencePromptModelMetadata,
                RunInferenceRunReady, RunInferenceServerMessage,
            },
        },
        transport::{DetachedTransport, TransportHandle, TransportRegistry},
        TransportCommand,
    },
    model::{
        image_from_encoded, is_image_mime, ContentPart, ModelInvocation, ModelLatency,
        ModelRequest, ModelSpec, ProjectedContent, ProjectedMessage, PromptSpec, ReasoningSpec,
        Role, ToolCallContent, ToolResultContent,
    },
    provider::{ModelEvent, Provider},
    Error, Result,
};

const ROLE_USER: i32 = 1;
const ROLE_ASSISTANT: i32 = 2;
const ROLE_TOOL: i32 = 3;
const ROLE_SYSTEM: i32 = 4;
const INVOCATION_ERROR_INTERNAL: i32 = 13;

#[derive(Clone)]
pub struct Session {
    pub model_hash: String,
    pub conversation_id: String,
    pub reasoning_effort: Option<String>,
    pub context_window_tokens: Option<u64>,
    pub fast: bool,
    pub cancel: CancellationToken,
    pub finished: CancellationToken,
    pub invocations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    pub calls: Arc<AtomicU64>,
}

pub enum Disposition {
    Handled,
    Upstream,
}

pub async fn accept(
    registry: &TransportRegistry,
    request_id: &str,
    message: inference::RunInferenceClientMessage,
) -> Result<Disposition> {
    let Some(message) = message.message else {
        return Ok(Disposition::Handled);
    };
    match message {
        run_inference_client_message::Message::RunRequest(request) => {
            begin_run(registry, request_id, request).await
        }
        run_inference_client_message::Message::InvokeModel(invoke) => {
            if registry.inference_session(request_id).await.is_none() {
                if registry.upstream(request_id).await {
                    return Ok(Disposition::Upstream);
                }
                return Err(Error::Protocol(
                    "RunInference invoke arrived before a local run_request".into(),
                ));
            }
            invoke_model(registry, request_id, invoke).await?;
            Ok(Disposition::Handled)
        }
        run_inference_client_message::Message::CancelInvocation(cancel) => {
            cancel_invocation(registry, request_id, &cancel.invocation_id).await;
            Ok(Disposition::Handled)
        }
        run_inference_client_message::Message::FinishRun(_) => {
            finish_run(registry, request_id).await;
            Ok(Disposition::Handled)
        }
        run_inference_client_message::Message::ClientHeartbeat(_) => Ok(Disposition::Handled),
    }
}

async fn begin_run(
    registry: &TransportRegistry,
    request_id: &str,
    request: inference::RunInferenceRunRequest,
) -> Result<Disposition> {
    let model = request.requested_model.ok_or_else(|| {
        Error::Protocol("RunInference run_request is missing requested_model".into())
    })?;
    let model_hash = model.model_id.clone();
    if model_hash.is_empty() {
        return Err(Error::Protocol(
            "RunInference run_request is missing a model id".into(),
        ));
    }
    if request.conversation_id.is_empty() {
        return Err(Error::Protocol(
            "RunInference run_request is missing a conversation id".into(),
        ));
    }
    if !model_is_local(registry, &model_hash).await? {
        tracing::info!(
            request_id,
            model_id = model_hash,
            "routing Cursor RunInference to Cursor upstream"
        );
        registry.mark_upstream(request_id).await;
        registry.trace(request_id).begin(
            Some(&request.conversation_id),
            "cursor_official",
            Some(&model_hash),
        );
        return Ok(Disposition::Upstream);
    }

    tracing::info!(
        request_id,
        conversation_id = request.conversation_id,
        model_id = model_hash,
        "routing Cursor RunInference to BYOK provider"
    );
    let (effort, context, fast) = parameters(&model.parameters);
    let metadata = prompt_metadata(effort.as_deref(), context);
    let session = Session {
        model_hash: model_hash.clone(),
        conversation_id: request.conversation_id.clone(),
        reasoning_effort: effort,
        context_window_tokens: context,
        fast,
        cancel: CancellationToken::new(),
        finished: CancellationToken::new(),
        invocations: Arc::new(Mutex::new(HashMap::new())),
        calls: Arc::new(AtomicU64::new(0)),
    };
    let detached = registry.open_detached(request_id).await?;
    detached
        .handle
        .set_conversation_id(&request.conversation_id)?;
    registry
        .insert_inference(request_id.to_string(), session.clone())
        .await;
    let handle = detached.handle.clone();
    watch_disconnect(detached, session.clone());
    emit(
        &handle,
        &RunInferenceServerMessage {
            message: Some(run_inference_server_message::Message::RunReady(
                RunInferenceRunReady {
                    resolved_model: Some(model),
                    prompt_model_metadata: Some(metadata),
                },
            )),
        },
    )?;
    registry.trace(request_id).begin(
        Some(&request.conversation_id),
        "local_byok",
        Some(&model_hash),
    );
    Ok(Disposition::Handled)
}

fn watch_disconnect(detached: DetachedTransport, session: Session) {
    let Some(mut commands) = detached.commands else {
        return;
    };
    let handle = detached.handle;
    tokio::spawn(async move {
        tokio::select! {
            biased;
            _ = session.finished.cancelled() => {}
            command = commands.recv() => {
                if matches!(command, Some(TransportCommand::Disconnect)) {
                    session.cancel.cancel();
                    handle.mark_disconnected();
                    handle.close_output();
                    handle.close_transport();
                }
            }
        }
    });
}

async fn invoke_model(
    registry: &TransportRegistry,
    request_id: &str,
    invoke: inference::RunInferenceInvokeModel,
) -> Result<()> {
    let session = registry
        .inference_session(request_id)
        .await
        .ok_or_else(|| Error::Protocol("RunInference session is gone".into()))?;
    let handle = registry
        .local(request_id)
        .await
        .ok_or_else(|| Error::RunNotFound(request_id.into()))?;
    let invocation_id = if invoke.invocation_id.is_empty() {
        format!("{request_id}:invoke")
    } else {
        invoke.invocation_id.clone()
    };
    let messages = invoke
        .request
        .map(|request| request.messages)
        .unwrap_or_default();
    let history = match project_messages(&messages) {
        Ok(history) if !history.is_empty() => history,
        Ok(_) => {
            emit_end(
                &handle,
                &invocation_id,
                Some("RunInference invoke has no messages".into()),
            )?;
            return Ok(());
        }
        Err(error) => {
            emit_end(&handle, &invocation_id, Some(error.to_string()))?;
            return Ok(());
        }
    };
    let images = history_image_count(&history);
    tracing::info!(
        request_id,
        invocation_id,
        messages = history.len(),
        images,
        "RunInference invoke forwarding to provider"
    );
    let cancel = session.cancel.child_token();
    session
        .invocations
        .lock()
        .await
        .insert(invocation_id.clone(), cancel.clone());
    let provider = registry.conversations().dependencies().provider.clone();
    let call_index = session.calls.fetch_add(1, Ordering::Relaxed);
    tokio::spawn(async move {
        let result = drive(
            provider,
            handle.clone(),
            session.clone(),
            invocation_id.clone(),
            history,
            cancel,
            call_index,
        )
        .await;
        session.invocations.lock().await.remove(&invocation_id);
        if let Err(error) = result {
            tracing::warn!(%error, %invocation_id, "RunInference provider stream failed");
            let _ = emit_end(&handle, &invocation_id, Some(error.to_string()));
        }
    });
    Ok(())
}

async fn drive(
    provider: Arc<dyn Provider>,
    handle: TransportHandle,
    session: Session,
    invocation_id: String,
    history: Vec<ProjectedMessage>,
    cancel: CancellationToken,
    call_index: u64,
) -> Result<()> {
    let mut model = ModelSpec::new(&session.model_hash);
    model.reasoning = ReasoningSpec {
        enabled: session.reasoning_effort.is_some(),
        effort: session.reasoning_effort.clone(),
    };
    model.latency = if session.fast {
        ModelLatency::Fast
    } else {
        ModelLatency::Standard
    };
    model.context_window_tokens = session.context_window_tokens;
    let invocation = ModelInvocation {
        call_id: format!("{}:{invocation_id}:{call_index}", handle.request_id()),
        run_id: handle.request_id().to_string(),
        conversation_id: session.conversation_id.clone(),
        provider_call_index: call_index,
        request: ModelRequest {
            prompt: PromptSpec {
                instructions: String::new(),
                tools: Vec::new(),
            },
            model,
            history,
        },
    };
    let mut stream = provider.stream(invocation, cancel.clone());
    while let Some(event) = stream.next().await {
        if cancel.is_cancelled() || session.finished.is_cancelled() {
            return Ok(());
        }
        match event? {
            ModelEvent::TextDelta(text) if !text.is_empty() => {
                emit_text(&handle, &invocation_id, &text, false)?;
            }
            ModelEvent::ThinkingDelta(text) if !text.is_empty() => {
                emit_thinking(&handle, &invocation_id, &text)?;
            }
            ModelEvent::Done(_) => break,
            _ => {}
        }
    }
    if cancel.is_cancelled() || session.finished.is_cancelled() {
        return Ok(());
    }
    emit_text(&handle, &invocation_id, "", true)?;
    emit_end(&handle, &invocation_id, None)?;
    Ok(())
}

async fn cancel_invocation(registry: &TransportRegistry, request_id: &str, invocation_id: &str) {
    let Some(session) = registry.inference_session(request_id).await else {
        return;
    };
    let cancel = session.invocations.lock().await.get(invocation_id).cloned();
    if let Some(cancel) = cancel {
        cancel.cancel();
    }
}

async fn finish_run(registry: &TransportRegistry, request_id: &str) {
    let Some(session) = registry.inference_session(request_id).await else {
        if let Some(handle) = registry.local(request_id).await {
            close_stream(&handle);
        }
        return;
    };
    session.cancel.cancel();
    session.finished.cancel();
    registry.remove_inference(request_id).await;
    if let Some(handle) = registry.local(request_id).await {
        close_stream(&handle);
    }
}

fn close_stream(handle: &TransportHandle) {
    if handle.emit_frame(connect::encode_end_stream()) {
        handle.close_output();
    }
    handle.close_transport();
}

async fn model_is_local(registry: &TransportRegistry, model_id: &str) -> Result<bool> {
    if model_id.starts_with(crate::plugin::ADAPTER_ID_PREFIX) {
        return Ok(true);
    }
    Ok(registry.store().model(model_id).await?.is_some())
}

fn prompt_metadata(
    reasoning_effort: Option<&str>,
    context_window_tokens: Option<u64>,
) -> RunInferencePromptModelMetadata {
    const MODES: &[(&str, bool)] = &[
        ("agent", true),
        ("chat", false),
        ("plan", false),
        ("debug", false),
        ("triage", false),
        ("multitask", false),
    ];
    RunInferencePromptModelMetadata {
        vendor: "unknown".into(),
        prompt_version: "latest".into(),
        reasoning_effort: reasoning_effort
            .filter(|effort| {
                matches!(
                    *effort,
                    "low" | "medium" | "high" | "extra-high" | "xhigh" | "max"
                )
            })
            .map(str::to_string),
        agent_token_limit: context_window_tokens.and_then(|tokens| i32::try_from(tokens).ok()),
        agent_mode_configs: MODES
            .iter()
            .map(|(mode_id, ask)| RunInferenceAgentModeConfig {
                mode_id: (*mode_id).into(),
                enable_ask_question_tool: *ask,
            })
            .collect(),
    }
}

fn parameters(values: &[InferenceModelParameterValue]) -> (Option<String>, Option<u64>, bool) {
    let mut effort = None;
    let mut context = None;
    let mut fast = false;
    for value in values {
        match value.id.as_str() {
            "reasoning" | "reasoning_effort" | "effort" if !value.value.is_empty() => {
                effort = Some(value.value.clone());
            }
            "context" | "context_window" => context = parse_context(&value.value),
            "fast" => fast = value.value.eq_ignore_ascii_case("true"),
            _ => {}
        }
    }
    (effort, context, fast)
}

fn parse_context(value: &str) -> Option<u64> {
    let value = value.trim();
    let (number, multiplier) = if let Some(number) = value.strip_suffix(['k', 'K']) {
        (number, 1_000.0)
    } else if let Some(number) = value.strip_suffix(['m', 'M']) {
        (number, 1_000_000.0)
    } else {
        (value, 1.0)
    };
    let number = number.trim().parse::<f64>().ok()?;
    let tokens = (number * multiplier) as u64;
    (tokens > 0).then_some(tokens)
}

pub(crate) fn project_messages(messages: &[InferenceCoreMessage]) -> Result<Vec<ProjectedMessage>> {
    let mut projected = Vec::new();
    for (index, message) in messages.iter().enumerate() {
        let message_id = format!("inference:{index}");
        match &message.content {
            Some(inference_core_message::Content::ToolContent(tool)) => {
                for (part_index, part) in tool.parts.iter().enumerate() {
                    projected.push(project_tool_part(
                        &format!("{message_id}:{part_index}"),
                        index,
                        part_index,
                        part,
                    )?);
                }
            }
            other => {
                let mut parts = Vec::new();
                match other {
                    Some(inference_core_message::Content::Text(text)) => {
                        parts.push(ContentPart::Text { text: text.clone() });
                    }
                    Some(inference_core_message::Content::Parts(group)) => {
                        for part in &group.parts {
                            push_content_part(&mut parts, part)?;
                        }
                    }
                    Some(inference_core_message::Content::ToolContent(_)) | None => {}
                }
                let calls = tool_calls(message)?;
                if parts.is_empty() && calls.is_empty() {
                    continue;
                }
                let has_image = parts
                    .iter()
                    .any(|part| matches!(part, ContentPart::Image { .. }));
                let role = role_of(message.role);
                if role == Role::Assistant && !calls.is_empty() && !has_image {
                    projected.push(ProjectedMessage {
                        message_id,
                        role,
                        content: ProjectedContent::Assistant {
                            text: join_text(&parts),
                            thinking: String::new(),
                            replay_state: None,
                            calls,
                        },
                    });
                } else if role == Role::Tool {
                    projected.push(ProjectedMessage {
                        message_id,
                        role,
                        content: ProjectedContent::ToolResult(ToolResultContent {
                            call_id: format!("tool-{index}"),
                            name: String::new(),
                            content: join_text(&parts),
                            is_error: false,
                            image: None,
                            provider_parts: parts
                                .into_iter()
                                .filter(|part| matches!(part, ContentPart::Image { .. }))
                                .collect(),
                        }),
                    });
                } else {
                    projected.push(ProjectedMessage {
                        message_id,
                        role,
                        content: ProjectedContent::Parts(parts),
                    });
                }
            }
        }
    }
    Ok(projected)
}

fn project_tool_part(
    message_id: &str,
    index: usize,
    part_index: usize,
    part: &inference::InferenceToolResultPart,
) -> Result<ProjectedMessage> {
    let mut parts = Vec::new();
    if let Some(text) = part
        .result
        .as_ref()
        .and_then(|value| value.string_value.clone())
    {
        parts.push(ContentPart::Text { text });
    }
    for content in &part.experimental_content {
        push_content_part(&mut parts, content)?;
    }
    let call_id = if part.tool_call_id.is_empty() {
        format!("tool-{index}-{part_index}")
    } else {
        part.tool_call_id.clone()
    };
    Ok(ProjectedMessage {
        message_id: message_id.into(),
        role: Role::Tool,
        content: ProjectedContent::ToolResult(ToolResultContent {
            call_id,
            name: part.tool_name.clone(),
            content: join_text(&parts),
            is_error: part.is_error,
            image: None,
            provider_parts: parts
                .into_iter()
                .filter(|part| matches!(part, ContentPart::Image { .. }))
                .collect(),
        }),
    })
}

fn push_content_part(parts: &mut Vec<ContentPart>, part: &InferenceContentPart) -> Result<()> {
    match &part.part {
        Some(inference_content_part::Part::Text(text)) => {
            parts.push(ContentPart::Text {
                text: text.text.clone(),
            });
        }
        Some(inference_content_part::Part::Image(image)) => {
            let (mime_type, data) = image_from_encoded(&image.data, image.mime_type.as_deref())?;
            parts.push(ContentPart::Image { mime_type, data });
        }
        Some(inference_content_part::Part::File(file)) if is_image_mime(&file.media_type) => {
            let (mime_type, data) = image_from_encoded(&file.data, Some(file.media_type.as_str()))?;
            parts.push(ContentPart::Image { mime_type, data });
        }
        Some(inference_content_part::Part::File(_)) | None => {}
    }
    Ok(())
}

fn tool_calls(message: &InferenceCoreMessage) -> Result<Vec<ToolCallContent>> {
    message
        .tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let arguments = call
                .raw_tool_call_args
                .as_deref()
                .filter(|args| !args.is_empty())
                .map(serde_json::from_str)
                .transpose()?
                .unwrap_or_else(|| serde_json::json!({}));
            Ok(ToolCallContent {
                index,
                call_id: call.tool_call_id.clone(),
                name: call.tool_name.clone(),
                arguments,
            })
        })
        .collect()
}

fn role_of(role: i32) -> Role {
    match role {
        ROLE_ASSISTANT => Role::Assistant,
        ROLE_TOOL => Role::Tool,
        ROLE_SYSTEM => Role::System,
        ROLE_USER => Role::User,
        _ => Role::User,
    }
}

fn join_text(parts: &[ContentPart]) -> String {
    let mut text = String::new();
    for part in parts {
        if let ContentPart::Text { text: part } = part {
            text.push_str(part);
        }
    }
    text
}

fn history_image_count(messages: &[ProjectedMessage]) -> usize {
    messages
        .iter()
        .map(|message| match &message.content {
            ProjectedContent::Parts(parts) => image_parts(parts),
            ProjectedContent::ToolResult(result) => image_parts(&result.provider_parts),
            ProjectedContent::Assistant { .. } => 0,
        })
        .sum()
}

fn image_parts(parts: &[ContentPart]) -> usize {
    parts
        .iter()
        .filter(|part| matches!(part, ContentPart::Image { .. }))
        .count()
}

fn emit(handle: &TransportHandle, message: &RunInferenceServerMessage) -> Result<()> {
    if handle.emit_frame(connect::encode_message(message)?) {
        Ok(())
    } else {
        Err(Error::RunNotFound(handle.request_id().to_string()))
    }
}

fn emit_text(
    handle: &TransportHandle,
    invocation_id: &str,
    text: &str,
    is_final: bool,
) -> Result<()> {
    emit(
        handle,
        &response(
            invocation_id,
            inference_stream_response::Response::TextPart(InferenceTextStreamPart {
                text: text.into(),
                is_final,
            }),
        ),
    )
}

fn emit_thinking(handle: &TransportHandle, invocation_id: &str, text: &str) -> Result<()> {
    emit(
        handle,
        &response(
            invocation_id,
            inference_stream_response::Response::ThinkingPart(InferenceThinkingStreamPart {
                text: text.into(),
            }),
        ),
    )
}

fn emit_end(handle: &TransportHandle, invocation_id: &str, error: Option<String>) -> Result<()> {
    emit(
        handle,
        &RunInferenceServerMessage {
            message: Some(run_inference_server_message::Message::InvocationEnd(
                RunInferenceInvocationEnd {
                    invocation_id: invocation_id.into(),
                    error: error.map(|message| RunInferenceInvocationError {
                        code: INVOCATION_ERROR_INTERNAL,
                        message,
                    }),
                },
            )),
        },
    )
}

fn response(
    invocation_id: &str,
    response: inference_stream_response::Response,
) -> RunInferenceServerMessage {
    RunInferenceServerMessage {
        message: Some(run_inference_server_message::Message::InvocationResponse(
            RunInferenceInvocationResponse {
                invocation_id: invocation_id.into(),
                response: Some(InferenceStreamResponse {
                    response: Some(response),
                }),
            },
        )),
    }
}

/// True when `payload` is a `RunInferenceClientMessage` whose run request starts
/// with a conversation id. Agent runs nest a `ConversationStateStructure` in
/// that position, so a UUID here is the 3.27 inference envelope.
pub(crate) fn is_run_request_payload(payload: &[u8]) -> bool {
    let Some(request) = length_delimited(payload, 1) else {
        return false;
    };
    let Some(conversation_id) = length_delimited(request, 1) else {
        return false;
    };
    std::str::from_utf8(conversation_id).is_ok_and(is_uuid)
}

fn is_uuid(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    let groups = [8, 4, 4, 4, 12];
    let mut offset = 0;
    for (index, group) in groups.iter().enumerate() {
        if index > 0 {
            if bytes.get(offset) != Some(&b'-') {
                return false;
            }
            offset += 1;
        }
        if !bytes
            .get(offset..offset + group)
            .is_some_and(|group| group.iter().all(|byte| byte.is_ascii_hexdigit()))
        {
            return false;
        }
        offset += group;
    }
    offset == bytes.len()
}

fn length_delimited(data: &[u8], field: u32) -> Option<&[u8]> {
    let mut index = 0;
    while index < data.len() {
        let key = read_varint(data, &mut index)?;
        let number = (key >> 3) as u32;
        let wire_type = key & 7;
        match wire_type {
            0 => {
                read_varint(data, &mut index)?;
            }
            1 => index += 8,
            2 => {
                let length = read_varint(data, &mut index)? as usize;
                let end = index.checked_add(length)?;
                if end > data.len() {
                    return None;
                }
                if number == field {
                    return Some(&data[index..end]);
                }
                index = end;
            }
            5 => index += 4,
            _ => return None,
        }
    }
    None
}

fn read_varint(data: &[u8], index: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    let mut shift = 0;
    while *index < data.len() && shift < 64 {
        let byte = data[*index];
        *index += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Some(value);
        }
        shift += 7;
    }
    None
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::{prompt_metadata, RunInferenceRunReady};

    #[test]
    fn prompt_metadata_uses_the_client_fallback_family() {
        let ready = RunInferenceRunReady {
            resolved_model: None,
            prompt_model_metadata: Some(prompt_metadata(Some("high"), Some(200_000))),
        };
        let decoded = RunInferenceRunReady::decode(ready.encode_to_vec().as_slice()).unwrap();
        let metadata = decoded.prompt_model_metadata.unwrap();
        assert_eq!(metadata.vendor, "unknown");
        assert_eq!(metadata.prompt_version, "latest");
        assert_eq!(metadata.reasoning_effort.as_deref(), Some("high"));
        assert_eq!(metadata.agent_token_limit, Some(200_000));
        assert!(metadata
            .agent_mode_configs
            .iter()
            .any(|mode| { mode.mode_id == "agent" && mode.enable_ask_question_tool }));
        assert!(metadata
            .agent_mode_configs
            .iter()
            .any(|mode| mode.mode_id == "plan" && !mode.enable_ask_question_tool));
    }

    #[test]
    fn unrecognized_reasoning_effort_is_omitted() {
        let metadata = prompt_metadata(Some("turbo"), None);
        assert!(metadata.reasoning_effort.is_none());
        assert!(metadata.agent_token_limit.is_none());
    }
}
