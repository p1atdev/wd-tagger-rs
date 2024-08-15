use anyhow::Result;

use crate::{
    config::ModelConfig, error::TaggerError, processor::ImagePreprocessor, tagger::TaggerModel,
    tags::LabelTags,
};

/// Pipeline for tagging images.
pub struct TaggingPipeline {
    pub model: TaggerModel,
    pub preprocessor: ImagePreprocessor,
    pub tags: LabelTags,
}

impl TaggingPipeline {
    pub fn from_pretrained(model_name: &str) -> Result<Self, TaggerError> {
        let model = TaggerModel::from_pretrained(&model_name)?;
        let config = ModelConfig::from_pretrained(&model_name)?;
        let preprocessor = ImagePreprocessor::from_config(&config)?;
        let tags = LabelTags::from_pretrained(model_name)?;

        Ok(Self {
            model,
            preprocessor,
            tags,
        })
    }

    // TODO: predict method
}
