use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(propagate_version = false)]
pub struct Cli {
    /// Input and output options
    #[command(flatten)]
    pub io: InputOutput,

    /// Model version
    #[command(subcommand)]
    pub model: Option<ModelVersion>,

    /// Inference device
    #[cfg(any(feature = "cuda", feature = "tensorrt"))]
    #[arg(short, long, default_value = "0")]
    pub devices: Vec<i32>,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ModelVersion {
    /// Use the tagger model of v2 series
    #[command(name = "--v2")]
    V2 {
        #[arg(default_value_t = V2Model::default())]
        model: V2Model,
    },
    /// Use the tagger model of v3 series
    #[command(name = "--v3")]
    V3 {
        #[arg(default_value_t = V3Model::default())]
        model: V3Model,
    },
    /// Use a custom model with the specified parameters
    #[command(name = "--custom")]
    Custom(CustomModel),
}

#[derive(Args, Clone, Debug)]
pub struct CustomModel {
    /// Repository id on Hugging Face
    #[arg(short, long)]
    pub repo_id: String,

    /// Model filename
    #[arg(short, long, default_value = "model.onnx")]
    pub model_file: String,

    /// Config filename
    #[arg(short, long, default_value = "config.json")]
    pub config_file: String,

    /// Tag list filename
    #[arg(short, long, default_value = "selected_tags.csv")]
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
pub enum V2Model {
    Vit,
    Moat,
    SwinV2,
    Convnext,
    ConvnextV2,
}

impl ModelPreset for V2Model {
    fn repo_id(&self) -> String {
        match self {
            V2Model::Vit => "SmilingWolf/wd-v1-4-vit-tagger-v2".to_string(),
            V2Model::Moat => "SmilingWolf/wd-v1-4-moat-tagger-v2".to_string(),
            V2Model::SwinV2 => "SmilingWolf/wd-v1-4-swinv2-tagger-v2".to_string(),
            V2Model::Convnext => "SmilingWolf/wd-v1-4-convnext-tagger-v2".to_string(),
            V2Model::ConvnextV2 => "SmilingWolf/wd-v1-4-convnextv2-tagger-v2".to_string(),
        }
    }

    fn default() -> Self {
        V2Model::SwinV2
    }
}

impl ToString for V2Model {
    fn to_string(&self) -> String {
        match self {
            V2Model::Vit => "vit".to_string(),
            V2Model::Moat => "moat".to_string(),
            V2Model::SwinV2 => "swin-v2".to_string(),
            V2Model::Convnext => "convnext".to_string(),
            V2Model::ConvnextV2 => "convnext-v2".to_string(),
        }
    }
}

#[derive(Args, Debug, Clone)]
#[group(required = false, multiple = false)]
pub struct InputOutput {
    /// Input path to a file or a folder
    pub input: String,

    /// Output path to a file or a folder
    #[arg(short, long)]
    pub output: Option<String>,

    /// Threshold for the prediction
    #[arg(short, long, default_value = "0.35")]
    pub threshold: f32,

    /// Use MCut Thresholding
    #[arg(long)]
    pub mcut: bool,
}
