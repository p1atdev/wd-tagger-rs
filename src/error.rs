use std::fmt::Display;

#[derive(Debug, Clone)]
pub enum TaggerError {
    /// Error around HuggingFace API
    Hf(String),
    /// Error around ONNX Runtime
    Ort(String),
    // Error around CUDA
    Cuda(String),
    /// Error around the processor
    Processor(String),
    /// Error around the tag
    Tag(String),
}

impl Display for TaggerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaggerError::Hf(message) => write!(f, "HuggingFace Error: {}", message),
            TaggerError::Ort(message) => write!(f, "ONNX Runtime Error: {}", message),
            TaggerError::Cuda(message) => write!(f, "CUDA Error: {}", message),
            TaggerError::Processor(message) => write!(f, "Processor Error: {}", message),
            TaggerError::Tag(message) => write!(f, "Tag Error: {}", message),
        }
    }
}
impl std::error::Error for TaggerError {}
