use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_classes: u32,
    pub num_features: u32,
    pub global_pool: String,
    pub model_args: ModelArgs,
    pub pretrained_cfg: PretrainedCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArgs {
    pub act_layer: String,
    pub img_size: u32,
    pub window_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedCfg {
    pub custom_load: bool,
    pub input_size: Vec<u32>,
    pub fixed_input_size: bool,
    pub interpolation: String,
    pub crop_pct: u32,
    pub crop_mode: String,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub num_classes: u32,
    pub pool_size: Option<String>,
    pub first_conv: Option<String>,
    pub classifier: Option<String>,
}
