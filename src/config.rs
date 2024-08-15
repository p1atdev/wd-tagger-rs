use crate::{
    error::TaggerError,
    file::{ConfigFile, HfFile},
};
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: String,
    pub num_classes: u32,
    pub num_features: u32,
    pub pretrained_cfg: PretrainedCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedCfg {
    pub input_size: Vec<u32>, // [channels, height, width]
    pub fixed_input_size: bool,
    pub num_classes: u32,
}

impl ModelConfig {
    pub fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let config_file = ConfigFile::new(&repo_id).get()?;
        let json = fs::read_to_string(config_file).map_err(|e| TaggerError::Io(e.to_string()))?;
        let config: ModelConfig =
            serde_json::from_str(&json).map_err(|e| TaggerError::Io(e.to_string()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::file::ConfigFile;
    use std::fs;

    #[test]
    fn test_load_model_config() {
        let config_file = ConfigFile::new("SmilingWolf/wd-swinv2-tagger-v3")
            .get()
            .unwrap();
        let json = fs::read_to_string(config_file).unwrap();
        let _config: ModelConfig = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_load_model_config_from_pretrained() {
        let _config = ModelConfig::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3").unwrap();
    }

    #[test]
    fn test_load_model_config_from_pretrained_many() {
        let repo_ids = vec![
            "SmilingWolf/wd-eva02-large-tagger-v3".to_string(),
            "SmilingWolf/wd-vit-large-tagger-v3".to_string(),
            "SmilingWolf/wd-v1-4-swinv2-tagger-v2".to_string(),
            "SmilingWolf/wd-vit-tagger-v3".to_string(),
            "SmilingWolf/wd-swinv2-tagger-v3".to_string(),
            "SmilingWolf/wd-convnext-tagger-v3".to_string(),
        ];

        for repo_id in repo_ids {
            let _config = ModelConfig::from_pretrained(&repo_id);
            assert!(_config.is_ok(), "{}", repo_id);
        }
    }
}
