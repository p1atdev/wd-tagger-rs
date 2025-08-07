use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(propagate_version = false)]
pub struct Cli {
    /// Model version
    #[command(subcommand)]
    pub model: ModelVersion,

    /// Inference device
    #[cfg(any(feature = "cuda", feature = "tensorrt"))]
    #[arg(short, long, default_value = "0")]
    pub devices: Vec<i32>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelVersion {
    /// Use the tagger model of v3 series
    #[command(name = "v3")]
    V3 {
        /// Input and output options
        #[command(flatten)]
        io: InputOutput,

        #[arg(long, default_value_t = V3Model::default())]
        model: V3Model,
    },
    /// Use a custom model with the specified parameters
    #[command(name = "custom")]
    Custom(CustomModel),
}

#[derive(Args, Clone, Debug)]
pub struct CustomModel {
    /// Input and output options
    #[command(flatten)]
    pub io: InputOutput,

    /// Repository id on Hugging Face
    #[arg(short, long)]
    pub repo_id: String,

    /// Model filename
    #[arg(long, default_value = "model.onnx")]
    pub model_file: String,

    /// Config filename
    #[arg(long, default_value = "config.json")]
    pub config_file: String,

    /// Tag list filename
    #[arg(long, default_value = "selected_tags.csv")]
    pub tags_file: String,
}

pub trait ModelPreset {
    fn repo_id(&self) -> String;
    fn default() -> Self;
}

#[derive(ValueEnum, Debug, Clone)]
pub enum V3Model {
    Vit,
    SwinV2,
    Convnext,
    VitLarge,
    Eva02Large,
}

impl ModelPreset for V3Model {
    fn repo_id(&self) -> String {
        match self {
            V3Model::Vit => "SmilingWolf/wd-vit-tagger-v3".to_string(),
            V3Model::SwinV2 => "SmilingWolf/wd-swinv2-tagger-v3".to_string(),
            V3Model::Convnext => "SmilingWolf/wd-convnext-tagger-v3".to_string(),
            V3Model::VitLarge => "SmilingWolf/wd-vit-large-tagger-v3".to_string(),
            V3Model::Eva02Large => "SmilingWolf/wd-eva02-large-tagger-v3".to_string(),
        }
    }

    fn default() -> Self {
        V3Model::SwinV2
    }
}

impl ToString for V3Model {
    fn to_string(&self) -> String {
        match self {
            V3Model::Vit => "vit".to_string(),
            V3Model::SwinV2 => "swin-v2".to_string(),
            V3Model::Convnext => "convnext".to_string(),
            V3Model::VitLarge => "vit-large".to_string(),
            V3Model::Eva02Large => "eva02-large".to_string(),
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
pub enum OutputFormat {
    Json,
    Jsonl,
    Caption,
}

#[derive(Args, Debug, Clone)]
#[group(required = false)]
pub struct InputOutput {
    /// Input path to a file or a folder#
    pub input: String,

    /// Output path to a file or a folder
    #[arg(short, long)]
    pub output: Option<String>,

    /// Threshold for the prediction
    #[arg(short, long, default_value = "0.35")]
    pub threshold: f32,

    /// Output format
    #[arg(short, long)]
    pub format: Option<OutputFormat>,

    /// Batch size for processing
    #[arg(short, long, default_value_t = 1)]
    pub batch_size: usize,
}
