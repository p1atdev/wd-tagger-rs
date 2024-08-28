use std::path::Path;

use anyhow::Result;
use ndarray::{Array, Axis, Ix4};
use ort::{CPUExecutionProvider, Session};

#[cfg(feature = "cuda")]
use ort::CUDAExecutionProvider;

#[cfg(feature = "tensorrt")]
use ort::TensorRTExecutionProvider;

use crate::error::TaggerError;
use crate::file::{HfFile, TaggerModelFile};

/// Enum for selecting the CUDA device
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    /// CUDA with default device
    #[cfg(feature = "cuda")]
    Cuda,
    /// CUDA with specific device
    #[cfg(feature = "cuda")]
    CudaDevice(i32),
    /// TensorRT
    #[cfg(feature = "tensorrt")]
    TensorRT,
    /// TensorRT with specific device
    #[cfg(feature = "tensorrt")]
    TensorRTDevice(i32),
}

/// Ailas for the device
impl Device {
    /// Use CPU
    pub fn cpu() -> Vec<Self> {
        vec![Self::Cpu]
    }

    /// Use CUDA with default device
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Vec<Self> {
        vec![Self::Cuda]
    }

    /// Use CUDA with specific device
    #[cfg(feature = "cuda")]
    pub fn cuda_devices(device_ids: Vec<i32>) -> Vec<Self> {
        device_ids.into_iter().map(Self::CudaDevice).collect()
    }

    /// Use TensorRT
    #[cfg(feature = "tensorrt")]
    pub fn tensorrt() -> Vec<Self> {
        vec![Self::TensorRT]
    }

    /// Use TensorRT with specific device
    #[cfg(feature = "tensorrt")]
    pub fn tensorrt_devices(device_ids: Vec<i32>) -> Vec<Self> {
        device_ids.into_iter().map(Self::TensorRTDevice).collect()
    }
}

/// Model for the Tagger
#[derive(Debug)]

pub struct TaggerModel {
    session: Session,
}

impl TaggerModel {
    /// Specify the devices to use
    pub fn use_devices(devices: Vec<Device>) -> Result<(), TaggerError> {
        match tracing_subscriber::fmt::try_init() {
            Ok(_) => {}
            Err(e) => println!("Warning: Failed to initialize the logger: {}", e),
        }

        let privders = devices
            .iter()
            .map(|device| match device {
                Device::Cpu => CPUExecutionProvider::default().build(),
                #[cfg(feature = "cuda")]
                Device::Cuda => CUDAExecutionProvider::default().build(),
                #[cfg(feature = "cuda")]
                Device::CudaDevice(device_id) => {
                    let provider = CUDAExecutionProvider::default();
                    provider.with_device_id(device_id.clone()).build()
                }
                #[cfg(feature = "tensorrt")]
                Device::TensorRT => TensorRTExecutionProvider::default().build(),
                #[cfg(feature = "tensorrt")]
                Device::TensorRTDevice(device_id) => {
                    let provider = TensorRTExecutionProvider::default();
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
            .commit_from_file(model_path)
            .map_err(|e| TaggerError::Ort(e.to_string()))?;

        Ok(Self { session })
    }

    /// Load the model in user-friendly way using the repo_id
    pub fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let model_path = TaggerModelFile::new(repo_id)
            .get()
            .map_err(|e| TaggerError::Hf(e.to_string()))?;

        Self::load(model_path)
    }

    pub fn predict(&self, input_tensor: Array<f32, Ix4>) -> Result<Vec<Vec<f32>>, TaggerError> {
        let inputs = ort::inputs![input_tensor].map_err(|e| TaggerError::Ort(e.to_string()))?;
        let output = self
            .session
            .run(inputs)
            .map_err(|e| TaggerError::Ort(e.to_string()))?;
        let preds = output["output"].try_extract_tensor::<f32>().unwrap();

        let preds = preds
            .axis_iter(Axis(0))
            .map(|row| row.iter().copied().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        Ok(preds)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::file::{HfFile, TaggerModelFile};
    use crate::processor::{ImagePreprocessor, ImageProcessor};
    use image;
    use ndarray::Axis;
    use ort::SessionOutputs;

    #[test]
    fn test_use_cpu() {
        let devices = vec![Device::Cpu];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_use_cuda_auto() {
        let devices = vec![Device::Cuda];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_use_cuda_device() {
        let devices = vec![Device::CudaDevice(0)];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    #[cfg(feature = "tensorrt")]
    fn test_use_tensorrt() {
        let devices = vec![Device::TensorRT];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    #[cfg(feature = "tensorrt")]
    fn test_use_tensorrt_device() {
        let devices = vec![Device::TensorRTDevice(0)];
        assert!(TaggerModel::use_devices(devices).is_ok());
    }

    #[test]
    fn test_load_tagger_model() {
        let model_path = TaggerModelFile::new("SmilingWolf/wd-swinv2-tagger-v3")
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
        let model_path = TaggerModelFile::new("SmilingWolf/wd-swinv2-tagger-v3")
            .get()
            .unwrap();

        let model = TaggerModel::load(model_path).unwrap();

        let image = image::open("assets/sample1_3x1024x1024.webp").unwrap();
        let processor = ImagePreprocessor::new(3, 448, 448);
        let tensor = processor.process(&image).unwrap();
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
        let preds = preds.first().unwrap();

        dbg!("Total", &preds.len());

        // max value
        let max = preds.iter().fold(0.0f32, |acc, &x| acc.max(x));
        dbg!("Max:", &max);

        // first 5 value pairs
        let pairs = preds.iter().take(5).collect::<Vec<_>>();
        dbg!("Pairs:", &pairs); // [general, sensitive, questionable, explicit, 1girl]
    }
}
