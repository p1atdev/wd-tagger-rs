use std::path::Path;

use anyhow::Result;
use ndarray::{Array, Ix4};
use ort::{CUDAExecutionProvider, GraphOptimizationLevel, Session};

use crate::error::TaggerError;
use crate::file::{HfFile, TaggerModelFile};

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

/// Model for the Tagger
pub struct TaggerModel {
    session: Session,
}

impl TaggerModel {
    /// Specify the devices to use
    pub fn use_devices(devices: Vec<Device>) -> Result<(), TaggerError> {
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

    /// Load the model directly using the local file path
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

    /// Load the model in user-friendly way using the repo_id
    pub fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let model_path = TaggerModelFile::new(repo_id.to_string())
            .get()
            .map_err(|e| TaggerError::Hf(e.to_string()))?;

        Self::load(model_path)
    }

    pub fn predict(&self, input_tensor: Array<f32, Ix4>) -> Result<()> {
        let inputs = ort::inputs![input_tensor].unwrap();
        let output = self.session.run(inputs).unwrap();
        let logits = output["output"].try_extract_tensor::<f32>().unwrap();

        // TODO: Implement the post-processing
        println!("{}", logits);

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::file::{HfFile, TaggerModelFile};
    use crate::processor::{ImagePreprocessor, ImageProcessor};
    use image;
    use ndarray::{s, Axis};
    use ort::SessionOutputs;

    #[test]
    fn test_use_cpu() {
        let devices = vec![Device::Cpu];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

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
    fn test_from_pretrained() {
        let _ = TaggerModel::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3").is_ok();
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

        let output: SessionOutputs = model.session.run(inputs).unwrap();
        let preds = output["output"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .into_owned();

        dbg!(&output);
        println!("{}", &preds);

        let preds = preds
            .axis_iter(Axis(0))
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        dbg!("Total", &preds.len());

        for pred in preds.iter() {
            dbg!("Len:", &pred.len());
        }
    }
}
