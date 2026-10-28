//! Resolves images referenced by Cursor user messages and conversation history.
use std::collections::HashSet;

use crate::{
    cursor::protocol::proto::agent::v1 as pb,
    model::{
        image_from_encoded, materialize_inline, CanonicalMessage, ContentPart, MessageContent,
        Origin, Role,
    },
    store::BlobId,
    Error, Result,
};

pub async fn parts(
    message: &pb::UserMessage,
    text: String,
    blobs: &crate::cursor::services::blob_sync::BlobSynchronizer,
) -> Result<Vec<ContentPart>> {
    let mut parts = vec![ContentPart::Text { text }];
    if let Some(context) = &message.selected_context {
        for image in &context.selected_images {
            let inline = image_data(image, blobs).await?;
            let (mime_type, data) = materialize_inline(inline, image.mime_type.trim())?;
            parts.push(ContentPart::Image { mime_type, data });
        }
    }
    Ok(parts)
}

/// Images already present on hydrated or newly compiled messages.
#[derive(Default)]
pub(super) struct ImageIndex {
    digests: HashSet<String>,
    ids: HashSet<String>,
}

pub(super) fn index_images(messages: &[CanonicalMessage], index: &mut ImageIndex) {
    for message in messages {
        index.ids.insert(message.message_id.clone());
        match &message.content {
            MessageContent::Parts { parts } => index_parts(parts, index),
            MessageContent::ToolResult(result) => index_parts(&result.provider_parts, index),
            MessageContent::Assistant { .. } => {}
        }
    }
}

/// Appends history images the checkpoint did not already carry.
///
/// Current Cursor sends prior-turn images on `ConversationHistory` as base64
/// or data URLs, separate from `SelectedImage` on the live user message. Those
/// parts are not in `root_prompt_messages_json` unless this server wrote them.
/// One stable runtime message keeps them in later checkpoints without rewriting
/// earlier roots.
pub(super) fn missing_history_message(
    index: &ImageIndex,
    history: &pb::ConversationHistory,
) -> Result<Option<CanonicalMessage>> {
    let mut text = String::new();
    let mut images = Vec::new();
    for message in &history.messages {
        match message.message.as_ref() {
            Some(pb::conversation_history_message::Message::User(user)) => {
                push_history_user(user, index, &mut text, &mut images)?;
            }
            Some(pb::conversation_history_message::Message::Tool(tool)) => {
                push_history_tool(tool, index, &mut text, &mut images)?;
            }
            _ => {}
        }
    }
    if images.is_empty() {
        return Ok(None);
    }
    let mut parts = Vec::with_capacity(images.len() + 1);
    if !text.is_empty() {
        parts.push(ContentPart::Text { text });
    }
    parts.extend(images);
    let identity = BlobId::digest(&serde_json::to_vec(&parts)?).to_base64();
    let message_id = format!("runtime:history-images:{identity}");
    if index.ids.contains(&message_id) {
        return Ok(None);
    }
    Ok(Some(CanonicalMessage {
        message_id,
        role: Role::User,
        origin: Origin::Runtime,
        content: MessageContent::Parts { parts },
        runtime_event_id: Some(format!("history-images:{identity}")),
    }))
}

fn push_history_user(
    user: &pb::ConversationHistoryUserMessage,
    index: &ImageIndex,
    text: &mut String,
    images: &mut Vec<ContentPart>,
) -> Result<()> {
    let label = user
        .content
        .iter()
        .filter_map(|item| match item.content.as_ref() {
            Some(pb::conversation_history_user_content::Content::Text(text)) => {
                Some(text.text.as_str())
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut added = false;
    for item in &user.content {
        let Some(pb::conversation_history_user_content::Content::Image(image)) =
            item.content.as_ref()
        else {
            continue;
        };
        let part = history_image(image)?;
        if remember_image(&part, index, images) {
            added = true;
        }
    }
    if added && !label.trim().is_empty() {
        text.push_str("<history_image>\n<user_query>");
        text.push_str(label.trim());
        text.push_str("</user_query>\n</history_image>\n");
    }
    Ok(())
}

fn push_history_tool(
    tool: &pb::ConversationHistoryToolMessage,
    index: &ImageIndex,
    text: &mut String,
    images: &mut Vec<ContentPart>,
) -> Result<()> {
    let mut added = false;
    for item in &tool.content {
        let Some(pb::conversation_history_tool_result_content::Content::Image(image)) =
            item.content.as_ref()
        else {
            continue;
        };
        let part = history_image(image)?;
        if remember_image(&part, index, images) {
            added = true;
        }
    }
    if added {
        text.push_str("<history_image>\n<tool_result name=\"");
        text.push_str(&tool.tool_name);
        text.push_str("\" call_id=\"");
        text.push_str(&tool.tool_call_id);
        text.push_str("\"/>\n</history_image>\n");
    }
    Ok(())
}

fn history_image(image: &pb::ConversationHistoryImageContent) -> Result<ContentPart> {
    let (mime_type, data) = image_from_encoded(&image.data, image.mime_type.as_deref())?;
    Ok(ContentPart::Image { mime_type, data })
}

fn remember_image(part: &ContentPart, index: &ImageIndex, images: &mut Vec<ContentPart>) -> bool {
    let ContentPart::Image { data, .. } = part else {
        return false;
    };
    let digest = BlobId::digest(data).to_base64();
    if index.digests.contains(&digest)
        || images
            .iter()
            .any(|existing| image_digest(existing).as_ref() == Some(&digest))
    {
        return false;
    }
    images.push(part.clone());
    true
}

fn index_parts(parts: &[ContentPart], index: &mut ImageIndex) {
    for part in parts {
        if let Some(digest) = image_digest(part) {
            index.digests.insert(digest);
        }
    }
}

fn image_digest(part: &ContentPart) -> Option<String> {
    match part {
        ContentPart::Image { data, .. } => Some(BlobId::digest(data).to_base64()),
        ContentPart::Text { .. } => None,
    }
}

async fn image_data(
    image: &pb::SelectedImage,
    blobs: &crate::cursor::services::blob_sync::BlobSynchronizer,
) -> Result<Vec<u8>> {
    use pb::selected_image::DataOrBlobId;

    let data = match image.data_or_blob_id.as_ref() {
        Some(DataOrBlobId::Data(data)) => data.clone(),
        Some(DataOrBlobId::BlobId(raw_id)) => {
            let id = BlobId::from_bytes(raw_id)?;
            blobs.get(&id).await?.ok_or_else(|| {
                Error::Protocol(format!(
                    "selected image Blob is missing: {}",
                    id.to_base64()
                ))
            })?
        }
        Some(DataOrBlobId::BlobIdWithData(value)) => {
            let id = BlobId::from_bytes(&value.blob_id)?;
            if value.data.is_empty() {
                blobs.get(&id).await?.ok_or_else(|| {
                    Error::Protocol(format!(
                        "selected image Blob is missing: {}",
                        id.to_base64()
                    ))
                })?
            } else {
                blobs.cache_received(&id, &value.data).await?;
                value.data.clone()
            }
        }
        None => {
            return Err(Error::Protocol(
                "selected image is missing data_or_blob_id".into(),
            ))
        }
    };
    if data.is_empty() {
        return Err(Error::Protocol("selected image data is empty".into()));
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn history_with_image(data: &str) -> pb::ConversationHistory {
        pb::ConversationHistory {
            messages: vec![pb::ConversationHistoryMessage {
                message: Some(pb::conversation_history_message::Message::User(
                    pb::ConversationHistoryUserMessage {
                        content: vec![
                            pb::ConversationHistoryUserContent {
                                content: Some(
                                    pb::conversation_history_user_content::Content::Text(
                                        pb::ConversationHistoryTextContent {
                                            text: "what is in this screenshot?".into(),
                                        },
                                    ),
                                ),
                            },
                            pb::ConversationHistoryUserContent {
                                content: Some(
                                    pb::conversation_history_user_content::Content::Image(
                                        pb::ConversationHistoryImageContent {
                                            data: data.into(),
                                            mime_type: Some("image/png".into()),
                                        },
                                    ),
                                ),
                            },
                        ],
                    },
                )),
            }],
            replace_user_info: None,
        }
    }

    #[test]
    fn conversation_history_images_become_a_stable_user_message() {
        let history = history_with_image("data:image/png;base64,AQID");
        let message = missing_history_message(&ImageIndex::default(), &history)
            .unwrap()
            .unwrap();
        let MessageContent::Parts { parts } = &message.content else {
            panic!("history images must be message parts");
        };
        assert!(matches!(
            &parts[1],
            ContentPart::Image { mime_type, data }
                if mime_type == "image/png" && data == &vec![1, 2, 3]
        ));
        let mut index = ImageIndex::default();
        index_images(std::slice::from_ref(&message), &mut index);
        assert!(missing_history_message(&index, &history).unwrap().is_none());
    }

    #[test]
    fn invalid_history_mime_is_rejected() {
        let history = history_with_image("data:text/plain;base64,AQID");
        let error = missing_history_message(&ImageIndex::default(), &history).unwrap_err();
        assert!(error.to_string().contains("MIME"), "{error}");
    }
}
