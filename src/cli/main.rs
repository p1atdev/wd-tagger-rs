mod args;
mod file;

use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use args::{Cli, ModelPreset, ModelVersion, OutputFormat};
use clap::Parser;
use wdtagger::{
    config::ModelConfig,
    file::{ConfigFile, HfFile, TagCSVFile, TaggerModelFile},
    pipeline::TaggingPipeline,
    processor::ImagePreprocessor,
    tagger::{Device, TaggerModel},
    tags::LabelTags,
};

/// Get the target device type.
fn target_device_type() -> String {
    if cfg!(feature = "tensorrt") {
        "TensorRT".to_string()
    } else if cfg!(feature = "cuda") {
        "CUDA".to_string()
    } else if cfg!(feature = "coreml") {
        "CoreML".to_string()
    } else {
        "CPU".to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let target_device = target_device_type();
    println!("Target device: <{}>", target_device);

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

    let repo_id = match &cli.model {
        ModelVersion::V2 { model, .. } => model.repo_id(),
        ModelVersion::V3 { model, .. } => model.repo_id(),
        ModelVersion::Custom(custom) => custom.repo_id.clone(),
        // None => V3Model::default().repo_id(),
    };
    let model_file = match &cli.model {
        ModelVersion::Custom(custom) => custom.model_file.clone(),
        _ => "model.onnx".to_string(),
    };
    let config_file = match &cli.model {
        ModelVersion::Custom(custom) => custom.config_file.clone(),
        _ => "config.json".to_string(),
    };
    let tag_csv_file = match &cli.model {
        ModelVersion::Custom(custom) => custom.tags_file.clone(),
        _ => "selected_tags.csv".to_string(),
    };
    let io = match &cli.model {
        ModelVersion::V2 { io, .. } => io,
        ModelVersion::V3 { io, .. } => io,
        ModelVersion::Custom(custom) => &custom.io,
    };

    // define files
    let model_file = TaggerModelFile::custom(&repo_id, None, &model_file);
    let config_file = ConfigFile::custom(&repo_id, None, &config_file);
    let tag_csv_file = TagCSVFile::custom(&repo_id, None, &tag_csv_file);

    // pre-download files
    let model_file_path = model_file.get()?;
    let config_file_path = config_file.get()?;
    let tag_csv_file_path = tag_csv_file.get()?;

    // load model
    TaggerModel::use_devices(device)?; // do once
    let model = TaggerModel::load(&model_file_path)?;
    let config = ModelConfig::load(&config_file_path)?;
    let preprocessor = ImagePreprocessor::from_config(&config)?;
    let label_tags = LabelTags::load(&tag_csv_file_path)?;

    // I/O
    let input = PathBuf::from_str(&io.input)?;
    let output = &io.output;
    let format = if output.is_some() && io.format.is_none() {
        &Some(OutputFormat::Json)
    } else {
        &io.format
    };

    // load pipe
    let threshold = match format {
        Some(OutputFormat::Json) | Some(OutputFormat::Jsonl) => 0f32, // save all predictions
        Some(OutputFormat::Caption) | None => io.threshold, // keep predictions above threshold
    };
    let pipe = TaggingPipeline::new(model, preprocessor, label_tags, &threshold);

    // if input is single file
    match file::is_file(&input).await? {
        true => {
            let img = image::open(&input)?;
            let result = pipe.predict(img)?;
            dbg!(&result);

            match format {
                Some(OutputFormat::Json) => {
                    let save_path = if let Some(output) = output {
                        PathBuf::from_str(output)?
                    } else {
                        file::get_path_with_extension(input, "json")
                    };
                    println!("Saving result to: {save_path:?}");

                    file::write_as_json(&save_path, &result).await?;
                }
                Some(OutputFormat::Jsonl) => unimplemented!("Jsonl output is not implemented yet"),
                Some(OutputFormat::Caption) => {
                    let save_path = if let Some(output) = output {
                        PathBuf::from_str(output)?
                    } else {
                        file::get_path_with_extension(input, "txt")
                    };
                    println!("Saving result to: {save_path:?}");

                    file::write_as_caption(&save_path, &result).await?;
                }
                None => {} // do nothing
            };
        }
        false => {
            unimplemented!("Folder input is not implemented yet");
        }
    }

    Ok(())
}
