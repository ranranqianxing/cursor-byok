//! Wire subset of Cursor 3.27 `RunInference` messages.
//!
//! Prost skips fields that are not declared here, so later client fields do not
//! need a schema bump. Declared fields match the agent-host descriptors.

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceClientMessage {
    #[prost(
        oneof = "run_inference_client_message::Message",
        tags = "1, 2, 3, 4, 5"
    )]
    pub message: ::core::option::Option<run_inference_client_message::Message>,
}

pub mod run_inference_client_message {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Message {
        #[prost(message, tag = "1")]
        RunRequest(super::RunInferenceRunRequest),
        #[prost(message, tag = "2")]
        InvokeModel(super::RunInferenceInvokeModel),
        #[prost(message, tag = "3")]
        CancelInvocation(super::RunInferenceCancelInvocation),
        #[prost(message, tag = "4")]
        FinishRun(super::RunInferenceFinishRun),
        #[prost(message, tag = "5")]
        ClientHeartbeat(super::RunInferenceClientHeartbeat),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceRunRequest {
    #[prost(string, tag = "1")]
    pub conversation_id: String,
    #[prost(message, optional, tag = "3")]
    pub requested_model: ::core::option::Option<InferenceRequestedModel>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceRequestedModel {
    #[prost(string, tag = "1")]
    pub model_id: String,
    #[prost(message, repeated, tag = "3")]
    pub parameters: ::prost::alloc::vec::Vec<InferenceModelParameterValue>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceModelParameterValue {
    #[prost(string, tag = "1")]
    pub id: String,
    #[prost(string, tag = "2")]
    pub value: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceInvokeModel {
    #[prost(string, tag = "1")]
    pub invocation_id: String,
    #[prost(message, optional, tag = "2")]
    pub request: ::core::option::Option<InferenceStreamRequest>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceStreamRequest {
    #[prost(message, repeated, tag = "1")]
    pub messages: ::prost::alloc::vec::Vec<InferenceCoreMessage>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceCoreMessage {
    #[prost(int32, tag = "1")]
    pub role: i32,
    #[prost(oneof = "inference_core_message::Content", tags = "2, 3, 6")]
    pub content: ::core::option::Option<inference_core_message::Content>,
    #[prost(message, repeated, tag = "4")]
    pub tool_calls: ::prost::alloc::vec::Vec<InferenceToolCall>,
}

pub mod inference_core_message {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Content {
        #[prost(string, tag = "2")]
        Text(String),
        #[prost(message, tag = "3")]
        Parts(super::InferenceContentParts),
        #[prost(message, tag = "6")]
        ToolContent(super::InferenceToolResultContent),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceContentParts {
    #[prost(message, repeated, tag = "1")]
    pub parts: ::prost::alloc::vec::Vec<InferenceContentPart>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceContentPart {
    #[prost(oneof = "inference_content_part::Part", tags = "1, 2, 3")]
    pub part: ::core::option::Option<inference_content_part::Part>,
}

pub mod inference_content_part {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Part {
        #[prost(message, tag = "1")]
        Text(super::InferenceTextPart),
        #[prost(message, tag = "2")]
        Image(super::InferenceImagePart),
        #[prost(message, tag = "3")]
        File(super::InferenceFilePart),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceTextPart {
    #[prost(string, tag = "1")]
    pub text: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceImagePart {
    #[prost(string, tag = "1")]
    pub data: String,
    #[prost(string, optional, tag = "2")]
    pub mime_type: ::core::option::Option<String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceFilePart {
    #[prost(string, tag = "1")]
    pub data: String,
    #[prost(string, tag = "2")]
    pub media_type: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceToolCall {
    #[prost(string, tag = "1")]
    pub tool_call_id: String,
    #[prost(string, tag = "2")]
    pub tool_name: String,
    #[prost(string, optional, tag = "4")]
    pub raw_tool_call_args: ::core::option::Option<String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceToolResultContent {
    #[prost(message, repeated, tag = "1")]
    pub parts: ::prost::alloc::vec::Vec<InferenceToolResultPart>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceToolResultPart {
    #[prost(string, tag = "1")]
    pub tool_call_id: String,
    #[prost(string, tag = "2")]
    pub tool_name: String,
    /// `google.protobuf.Value`. Only the string arm is consumed; other arms are skipped.
    #[prost(message, optional, tag = "3")]
    pub result: ::core::option::Option<InferenceStringValue>,
    #[prost(bool, tag = "4")]
    pub is_error: bool,
    #[prost(message, repeated, tag = "5")]
    pub experimental_content: ::prost::alloc::vec::Vec<InferenceContentPart>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceStringValue {
    #[prost(string, optional, tag = "3")]
    pub string_value: ::core::option::Option<String>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceCancelInvocation {
    #[prost(string, tag = "1")]
    pub invocation_id: String,
}

#[derive(Clone, Copy, PartialEq, ::prost::Message)]
pub struct RunInferenceFinishRun {}

#[derive(Clone, Copy, PartialEq, ::prost::Message)]
pub struct RunInferenceClientHeartbeat {}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceServerMessage {
    #[prost(oneof = "run_inference_server_message::Message", tags = "2, 3, 4")]
    pub message: ::core::option::Option<run_inference_server_message::Message>,
}

pub mod run_inference_server_message {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Message {
        #[prost(message, tag = "2")]
        RunReady(super::RunInferenceRunReady),
        #[prost(message, tag = "3")]
        InvocationResponse(super::RunInferenceInvocationResponse),
        #[prost(message, tag = "4")]
        InvocationEnd(super::RunInferenceInvocationEnd),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceRunReady {
    #[prost(message, optional, tag = "1")]
    pub resolved_model: ::core::option::Option<InferenceRequestedModel>,
    /// Cursor 3.27's local loop will not build the invoke request without this.
    #[prost(message, optional, tag = "4")]
    pub prompt_model_metadata: ::core::option::Option<RunInferencePromptModelMetadata>,
}

/// Subset of `RunInferencePromptModelMetadata`.
///
/// `vendor` and `prompt_version` must be values this Cursor build already
/// recognizes. `unknown` + `latest` is the client's own fallback for a model
/// it has no family for. `use_dsv3_harness` stays unset so the loop does not
/// require a Composer persona.
#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferencePromptModelMetadata {
    #[prost(string, tag = "1")]
    pub vendor: String,
    #[prost(string, tag = "2")]
    pub prompt_version: String,
    #[prost(string, optional, tag = "30")]
    pub reasoning_effort: ::core::option::Option<String>,
    #[prost(int32, optional, tag = "32")]
    pub agent_token_limit: ::core::option::Option<i32>,
    #[prost(message, repeated, tag = "46")]
    pub agent_mode_configs: ::prost::alloc::vec::Vec<RunInferenceAgentModeConfig>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceAgentModeConfig {
    #[prost(string, tag = "1")]
    pub mode_id: String,
    #[prost(bool, tag = "2")]
    pub enable_ask_question_tool: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceInvocationResponse {
    #[prost(string, tag = "1")]
    pub invocation_id: String,
    #[prost(message, optional, tag = "2")]
    pub response: ::core::option::Option<InferenceStreamResponse>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceStreamResponse {
    #[prost(oneof = "inference_stream_response::Response", tags = "1, 9")]
    pub response: ::core::option::Option<inference_stream_response::Response>,
}

pub mod inference_stream_response {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Response {
        #[prost(message, tag = "1")]
        TextPart(super::InferenceTextStreamPart),
        #[prost(message, tag = "9")]
        ThinkingPart(super::InferenceThinkingStreamPart),
    }
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceTextStreamPart {
    #[prost(string, tag = "1")]
    pub text: String,
    #[prost(bool, tag = "2")]
    pub is_final: bool,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct InferenceThinkingStreamPart {
    #[prost(string, tag = "1")]
    pub text: String,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceInvocationEnd {
    #[prost(string, tag = "1")]
    pub invocation_id: String,
    #[prost(message, optional, tag = "2")]
    pub error: ::core::option::Option<RunInferenceInvocationError>,
}

#[derive(Clone, PartialEq, ::prost::Message)]
pub struct RunInferenceInvocationError {
    #[prost(int32, tag = "1")]
    pub code: i32,
    #[prost(string, tag = "2")]
    pub message: String,
}
