//! Accepts ordered Cursor Bidi append requests and routes them by request_id.
use prost::Message;

use crate::{
    cursor::{
        conversation::TransportCommand,
        protocol::{
            events,
            proto::{agent::v1 as agent, aiserver::v1 as ai},
        },
        transport::{TransportParent, TransportRegistry},
    },
    Error, Result,
};

pub struct DecodedAppend {
    pub request_id: String,
    pub seqno: i64,
    pub message: agent::AgentClientMessage,
}

pub enum DecodedBidi {
    Agent(Box<DecodedAppend>),
    Inference(InferenceDecoded),
}

pub struct InferenceDecoded {
    pub request_id: String,
    pub seqno: i64,
    pub message: crate::cursor::protocol::inference::RunInferenceClientMessage,
}

impl DecodedAppend {
    pub fn model_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str())
            .filter(|model| !model.is_empty())
            .or_else(|| {
                request
                    .model_details
                    .as_ref()
                    .map(|model| model.model_id.as_str())
                    .filter(|model| !model.is_empty())
            })
    }

    pub fn conversation_id(&self) -> Option<&str> {
        let agent::agent_client_message::Message::RunRequest(request) =
            self.message.message.as_ref()?
        else {
            return None;
        };
        request.conversation_id.as_deref()
    }

    pub fn is_background_task_completion(&self) -> bool {
        let Some(agent::agent_client_message::Message::RunRequest(request)) =
            self.message.message.as_ref()
        else {
            return false;
        };
        matches!(
            request
                .action
                .as_ref()
                .and_then(|action| action.action.as_ref()),
            Some(agent::conversation_action::Action::BackgroundTaskCompletionAction(_))
        )
    }

    pub fn trace_metadata(&self) -> serde_json::Value {
        let Some(message) = self.message.message.as_ref() else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": "empty",
            });
        };
        let agent::agent_client_message::Message::RunRequest(request) = message else {
            return serde_json::json!({
                "append_seqno": self.seqno,
                "message_type": client_message_type(message),
            });
        };
        let (action_type, history_messages, history_images) = request
            .action
            .as_ref()
            .and_then(|action| action.action.as_ref())
            .map(|action| match action {
                agent::conversation_action::Action::UserMessageAction(action) => {
                    let history = action.conversation_history.as_ref();
                    (
                        "user_message",
                        history.map_or(0, |history| history.messages.len()),
                        history.map_or(0, history_image_count),
                    )
                }
                agent::conversation_action::Action::BackgroundTaskCompletionAction(_) => {
                    ("background_task_completion", 0, 0)
                }
                agent::conversation_action::Action::ExecutePlanAction(_) => ("execute_plan", 0, 0),
                agent::conversation_action::Action::SummarizeAction(_) => ("summarize", 0, 0),
                _ => ("other", 0, 0),
            })
            .unwrap_or(("none", 0, 0));
        let state = request.conversation_state.as_ref();
        serde_json::json!({
            "append_seqno": self.seqno,
            "message_type": "run_request",
            "conversation_id": request.conversation_id,
            "model_id": self.model_id(),
            "action_type": action_type,
            "conversation_history_messages": history_messages,
            "conversation_history_images": history_images,
            "root_message_count": state.map_or(0, |state| state.root_prompt_messages_json.len()),
            "turn_count": state.map_or(0, |state| state.turns.len()),
            "prefetched_blob_count": request.pre_fetched_blobs.len(),
        })
    }
}

fn client_message_type(message: &agent::agent_client_message::Message) -> &'static str {
    use agent::agent_client_message::Message;
    match message {
        Message::RunRequest(_) => "run_request",
        Message::ExecClientMessage(_) => "exec_client_message",
        Message::ExecClientControlMessage(_) => "exec_client_control_message",
        Message::KvClientMessage(_) => "kv_client_message",
        Message::ConversationAction(_) => "conversation_action",
        Message::InteractionResponse(_) => "interaction_response",
        Message::ClientHeartbeat(_) => "client_heartbeat",
        Message::PrewarmRequest(_) => "prewarm_request",
    }
}

fn history_image_count(history: &agent::ConversationHistory) -> usize {
    use agent::{
        conversation_history_message::Message,
        conversation_history_tool_result_content::Content as ToolContent,
        conversation_history_user_content::Content as UserContent,
    };
    history
        .messages
        .iter()
        .map(|message| match message.message.as_ref() {
            Some(Message::User(user)) => user
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(UserContent::Image(_))))
                .count(),
            Some(Message::Tool(tool)) => tool
                .content
                .iter()
                .filter(|content| matches!(content.content, Some(ToolContent::Image(_))))
                .count(),
            _ => 0,
        })
        .sum()
}

pub fn decode(request: &ai::BidiAppendRequest, known_inference: bool) -> Result<DecodedBidi> {
    let request_id = request
        .request_id
        .as_ref()
        .map(|id| id.request_id.as_str())
        .filter(|id| !id.is_empty())
        .ok_or_else(|| Error::Protocol("BidiAppend request_id is required".into()))?;
    if !request.data_binary.is_empty() {
        return Err(Error::Protocol(
            "BidiAppend data_binary is not part of the captured protocol".into(),
        ));
    }
    if request.data.is_empty() {
        return Err(Error::Protocol(
            "BidiAppend contains no AgentClientMessage".into(),
        ));
    }
    let payload = hex::decode(&request.data)
        .map_err(|error| Error::Protocol(format!("invalid BidiAppend hex: {error}")))?;
    if known_inference || crate::cursor::inference::is_run_request_payload(&payload) {
        return Ok(DecodedBidi::Inference(InferenceDecoded {
            request_id: request_id.into(),
            seqno: request.append_seqno,
            message: crate::cursor::protocol::inference::RunInferenceClientMessage::decode(
                payload.as_slice(),
            )?,
        }));
    }
    match agent::AgentClientMessage::decode(payload.as_slice()) {
        Ok(message) => Ok(DecodedBidi::Agent(Box::new(DecodedAppend {
            request_id: request_id.into(),
            seqno: request.append_seqno,
            message,
        }))),
        Err(agent_error) => {
            match crate::cursor::protocol::inference::RunInferenceClientMessage::decode(
                payload.as_slice(),
            ) {
                Ok(message)
                    if message.message.as_ref().is_some_and(|message| {
                        !matches!(
                            message,
                            crate::cursor::protocol::inference::run_inference_client_message::Message::RunRequest(_)
                        )
                    }) =>
                {
                    Ok(DecodedBidi::Inference(InferenceDecoded {
                        request_id: request_id.into(),
                        seqno: request.append_seqno,
                        message,
                    }))
                }
                _ => Err(agent_error.into()),
            }
        }
    }
}

pub async fn append(
    registry: &TransportRegistry,
    request: DecodedAppend,
    parent: Option<TransportParent>,
) -> Result<ai::BidiAppendResponse> {
    let replace_closing = request.model_id().is_some();
    let handle = registry
        .get_or_create_for_append(&request.request_id, replace_closing)
        .await?;
    let _admission = handle.admit()?;
    if let Some(conversation_id) = request.conversation_id() {
        handle.set_conversation_id(conversation_id)?;
    }
    if let Some(parent) = parent {
        handle.set_parent(parent)?;
    }
    if matches!(
        request.message.message.as_ref(),
        Some(agent::agent_client_message::Message::ClientHeartbeat(_))
    ) {
        handle.emit(&events::heartbeat())?;
    }
    handle
        .command(TransportCommand::Append {
            seqno: request.seqno,
            message: Box::new(request.message),
        })
        .await?;
    Ok(ai::BidiAppendResponse {})
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::cursor::protocol::inference::{
        inference_content_part, inference_core_message, run_inference_client_message,
        InferenceContentPart, InferenceContentParts, InferenceCoreMessage, InferenceImagePart,
        InferenceRequestedModel, InferenceTextPart, RunInferenceClientMessage,
        RunInferenceInvokeModel, RunInferenceRunRequest,
    };

    fn append(payload: &[u8]) -> ai::BidiAppendRequest {
        ai::BidiAppendRequest {
            data: hex::encode(payload),
            request_id: Some(ai::BidiRequestId {
                request_id: "request-1".into(),
            }),
            append_seqno: 0,
            data_binary: Vec::new(),
        }
    }

    fn run_request(conversation_id: &str) -> Vec<u8> {
        let message = RunInferenceClientMessage {
            message: Some(run_inference_client_message::Message::RunRequest(
                RunInferenceRunRequest {
                    conversation_id: conversation_id.into(),
                    requested_model: Some(InferenceRequestedModel {
                        model_id: "8e1a15e48503a207".into(),
                        parameters: Vec::new(),
                    }),
                },
            )),
        };
        let mut payload = Vec::new();
        message.encode(&mut payload).unwrap();
        payload
    }

    #[test]
    fn run_inference_conversation_ids_do_not_decode_as_agent_state() {
        for conversation_id in [
            "5e7c3aa2-f7f2-46b1-a756-c60b1b98123c",
            "b6501dee-9d99-42e1-97a5-b781d6825c04",
            "e6501dee-9d99-42e1-97a5-b781d6825c04",
        ] {
            let payload = run_request(conversation_id);
            agent::AgentClientMessage::decode(payload.as_slice()).unwrap_err();
            let DecodedBidi::Inference(decoded) = decode(&append(&payload), false).unwrap() else {
                panic!("{conversation_id} was decoded as an agent run");
            };
            let run_inference_client_message::Message::RunRequest(request) =
                decoded.message.message.unwrap()
            else {
                panic!("{conversation_id} was not a run request");
            };
            assert_eq!(request.conversation_id, conversation_id);
            assert_eq!(
                request.requested_model.unwrap().model_id,
                "8e1a15e48503a207"
            );
        }
    }

    #[test]
    fn captured_cursor_3_27_run_request_decodes_as_inference() {
        let payload = include_bytes!("../../cursor/testdata/cursor_3_27_run_inference.bin");
        let DecodedBidi::Inference(decoded) = decode(&append(payload), false).unwrap() else {
            panic!("captured Cursor 3.27 append was decoded as an agent run");
        };
        let run_inference_client_message::Message::RunRequest(request) =
            decoded.message.message.unwrap()
        else {
            panic!("captured append was not a run request");
        };
        assert_eq!(
            request.conversation_id,
            "5e7c3aa2-f7f2-46b1-a756-c60b1b98123c"
        );
        let model = request.requested_model.unwrap();
        assert_eq!(model.model_id, "8e1a15e48503a207");
        assert!(model
            .parameters
            .iter()
            .any(|parameter| { parameter.id == "reasoning" && parameter.value == "high" }));
    }

    #[test]
    fn invoke_model_is_inference_when_agent_decode_fails() {
        let message = RunInferenceClientMessage {
            message: Some(run_inference_client_message::Message::InvokeModel(
                RunInferenceInvokeModel {
                    invocation_id: "invoke-1".into(),
                    request: Some(crate::cursor::protocol::inference::InferenceStreamRequest {
                        messages: vec![InferenceCoreMessage {
                            role: 1,
                            content: Some(inference_core_message::Content::Parts(
                                InferenceContentParts {
                                    parts: vec![
                                        InferenceContentPart {
                                            part: Some(inference_content_part::Part::Text(
                                                InferenceTextPart {
                                                    text: "What is the dominant color?".into(),
                                                },
                                            )),
                                        },
                                        InferenceContentPart {
                                            part: Some(inference_content_part::Part::Image(
                                                InferenceImagePart {
                                                    data: "data:image/png;base64,AQID".into(),
                                                    mime_type: None,
                                                },
                                            )),
                                        },
                                    ],
                                },
                            )),
                            tool_calls: Vec::new(),
                        }],
                    }),
                },
            )),
        };
        let mut payload = Vec::new();
        message.encode(&mut payload).unwrap();
        let DecodedBidi::Inference(decoded) = decode(&append(&payload), false).unwrap() else {
            panic!("invoke_model was decoded as an agent message");
        };
        let history = crate::cursor::inference::project_messages(match &decoded.message.message {
            Some(run_inference_client_message::Message::InvokeModel(invoke)) => {
                &invoke.request.as_ref().unwrap().messages
            }
            _ => panic!("expected invoke_model"),
        })
        .unwrap();
        let crate::model::ProjectedContent::Parts(parts) = &history[0].content else {
            panic!("user image turn was not projected as parts");
        };
        assert!(matches!(
            &parts[1],
            crate::model::ContentPart::Image { mime_type, data }
                if mime_type == "image/png" && data == &vec![1, 2, 3]
        ));
    }
}
