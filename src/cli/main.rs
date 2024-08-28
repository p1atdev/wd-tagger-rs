use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::fs::metadata;
use wdtagger::{pipeline::TaggingPipeline, tagger::Device};

#[derive(Parser, Debug, Clone)]
#[command(version, about, long_about = None)]
#[command(propagate_version = false)]
struct Cli {
    /// Input and output options
    #[command(flatten)]
    io: InputOutput,

    /// Model version
    #[command(subcommand)]
    model: Option<ModelVersion>,

    /// Inference device
    #[cfg(any(feature = "cuda", feature = "tensorrt"))]
    #[arg(short, long, default_value = "0")]
    devices: Vec<i32>,
}

#[derive(Debug, Clone, Subcommand)]
enum ModelVersion {
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
struct CustomModel {
    /// Repository id on Hugging Face
    #[arg(short, long)]
    repo_id: String,

    /// Model filename
    #[arg(short, long, default_value = "model.onnx")]
    model_file: String,

    /// Config filename
    #[arg(short, long, default_value = "config.json")]
    config_file: String,

    /// Tag list filename
    #[arg(short, long, default_value = "selected_tags.csv")]
    tags_file: String,
}

trait ModelPreset {
    fn repo_id(&self) -> String;
    fn default() -> Self;
}

#[derive(ValueEnum, Debug, Clone)]
enum V3Model {
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
enum V2Model {
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
struct InputOutput {
    /// Input path to a file or a folder
    input: String,

    /// Output path to a file or a folder
    #[arg(short, long)]
    output: Option<String>,

    /// Threshold for the prediction
    #[arg(short, long, default_value = "0.35")]
    threshold: f32,

    /// Use MCut Thresholding
    #[arg(long)]
    mcut: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let device = Device::cpu();

    #[cfg(feature = "cuda")]
    let device: Vec<Device> = cli.devices.iter().map(|d| Device::CudaDevice(*d)).collect();

    #[cfg(feature = "tensorrt")]
    let device: Vec<Device> = cli
        .devices
        .iter()
        .map(|d| Device::TensorRTDevice(*d))
        .collect();

    // load pipeline
    let pipe = match &cli.model {
        Some(ModelVersion::V2 { model }) => {
            TaggingPipeline::from_pretrained(&model.repo_id(), device)?
        }
        Some(ModelVersion::V3 { model }) => {
            TaggingPipeline::from_pretrained(&model.repo_id(), device)?
        }
        Some(ModelVersion::Custom(custom)) => {
            println!("Custom model: {:?}", custom);
            unimplemented!("Custom model is not implemented yet");
        }
        None => {
            let model = V3Model::default(); // use v3 default model
            TaggingPipeline::from_pretrained(&model.repo_id(), device)?
        }
    };

    // I/O
    let input = &cli.io.input;
    let output = &cli.io.output;
    let threshold = &cli.io.threshold;
    let mcut = &cli.io.mcut;

    // if input is single file
    match metadata(&input)?.is_file() {
        true => {
            let img = image::open(&input)?;
            let result = pipe.predict(img)?;
            dbg!(result);
        }
        false => {
            unimplemented!("Folder input is not implemented yet");
        }
    }

    dbg!(&cli);

    Ok(())
}
