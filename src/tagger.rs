use std::path::Path;

use anyhow::Result;
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};

use crate::error::TaggerError;

/// Enum for selecting the CUDA device
pub enum Device {
    Cpu,
    /// CUDA with default device
    #[cfg(feature = "cuda")]
    Cuda,
    /// CUDA with specific device
    #[cfg(feature = "cuda")]
    CudaDevice(i32),
}

/// ONNX Runtime Model Trait
pub trait OrtModel {
    /// Specify the devices to use
    fn use_devices(devices: Vec<Device>) -> Result<(), TaggerError> {
        use ort::CPUExecutionProvider;

        tracing_subscriber::fmt::init();

        let privders = devices
            .iter()
            .map(|device| match device {
                Device::Cpu => CPUExecutionProvider::default().build(),
                Device::Cuda => CUDAExecutionProvider::default().build(),
                Device::CudaDevice(device_id) => {
                    let provider = CUDAExecutionProvider::default();
                    provider.with_device_id(device_id.clone()).build()
                }
            })
            .collect::<Vec<_>>();

        match ort::init().with_execution_providers(privders).commit() {
            Ok(_) => Ok(()),
            Err(e) => Err(TaggerError::Cuda(e.to_string())),
        }
    }
}

/// Model for the Tagger
pub struct TaggerModel {
    session: Session,
}

impl TaggerModel {
    /// Initialize with the model_path
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self, TaggerError> {
        let builder = match Session::builder() {
            Ok(builder) => builder,
            Err(e) => return Err(TaggerError::Ort(e.to_string())),
        };

        let session = builder
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| TaggerError::Ort(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| TaggerError::Ort(e.to_string()))?;

        Ok(Self { session })
    }
}

impl OrtModel for TaggerModel {}

#[cfg(test)]
mod test {
    use super::*;
    use crate::file::{HfFile, TaggerModelFile};
    use crate::processor::{ImagePreprocessor, ImageProcessor};
    use image;

    #[test]
    fn test_use_cuda_auto() {
        let devices = vec![Device::Cuda];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    fn test_use_cuda_device() {
        let devices = vec![Device::CudaDevice(0)];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    fn test_load_tagger_model() {
        let model_path = TaggerModelFile::new("SmilingWolf/wd-swinv2-tagger-v3".to_string())
            .get()
            .unwrap();

        let _model = TaggerModel::load(model_path).unwrap();
    }

    #[test]
    fn test_run_tagger_model() {
        let model_path = TaggerModelFile::new("SmilingWolf/wd-swinv2-tagger-v3".to_string())
            .get()
            .unwrap();

        let model = TaggerModel::load(model_path).unwrap();

        let image = image::open("assets/sample1_3x448x448.webp").unwrap();
        let processor = ImagePreprocessor::new(3, 448, 448);
        let tensor = processor.process(image).unwrap();
        let inputs = ort::inputs![tensor].unwrap();

        let output = model.session.run(inputs).unwrap();
        let logits = output["output"].try_extract_tensor::<f32>().unwrap();

        println!("{}", logits);
    }
}
