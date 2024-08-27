use anyhow::Result;
use image::DynamicImage;
use indexmap::IndexMap;
use itertools::Itertools;

use crate::processor::{ImagePreprocessor, ImageProcessor};
use crate::tagger::Device;
use crate::tags::{LabelTags, TagCategory};
use crate::{config::ModelConfig, error::TaggerError, tagger::TaggerModel};

/// Pipeline for tagging images.
#[derive(Debug)]
pub struct TaggingPipeline {
    pub model: TaggerModel,
    pub preprocessor: ImagePreprocessor,
    pub tags: LabelTags,
    threshold: f32,
}

// type alias for prediction result
pub type Prediction = IndexMap<String, f32>;

fn sort_by_value(map: &Prediction) -> Prediction {
    map.into_iter()
        .sorted_by(|a, &b| b.1.partial_cmp(a.1).unwrap())
        .map(|(tag, prob)| (tag.clone(), *prob))
        .collect()
}

#[derive(Debug, Clone)]
pub struct TaggingResult {
    /// Rating tags
    pub rating: Prediction,
    /// Character tags
    pub character: Prediction,
    /// General tags
    pub general: Prediction,
}

impl TaggingResult {
    fn new(rating: &Prediction, character: &Prediction, general: &Prediction) -> Self {
        Self {
            rating: sort_by_value(&rating),
            character: sort_by_value(&character),
            general: sort_by_value(&general),
        }
    }
}

impl TaggingPipeline {
    /// Create a new tagging pipeline.
    pub fn from_pretrained(model_name: &str, devices: Vec<Device>) -> Result<Self, TaggerError> {
        TaggerModel::use_devices(devices)?;

        let model = TaggerModel::from_pretrained(&model_name)?;
        let config = ModelConfig::from_pretrained(&model_name)?;
        let preprocessor = ImagePreprocessor::from_config(&config)?;
        let tags = LabelTags::from_pretrained(model_name)?;

        Ok(Self {
            model,
            preprocessor,
            tags,
            threshold: 0.35,
        })
    }

    /// Predict the tags of an image.
    pub fn predict(&self, image: DynamicImage) -> Result<TaggingResult, TaggerError> {
        let tensor = self.preprocessor.process(&image)?;
        let probs = self.model.predict(tensor)?;
        let pairs = self.tags.create_probality_pairs(probs)?;
        let pairs = pairs.first().unwrap().clone();

        macro_rules! filter_tags {
            ($category:expr) => {
                pairs
                    .iter()
                    .filter(|(tag, &prob)| {
                        self.tags.label2tag().get(tag.clone()).unwrap().category() == $category
                            && &prob >= &self.threshold
                    })
                    .map(|(tag, prob)| (tag.clone(), *prob))
                    .collect::<Prediction>()
            };
        }
        let rating: Prediction = filter_tags!(TagCategory::Rating);
        let character: Prediction = filter_tags!(TagCategory::Character);
        let general: Prediction = filter_tags!(TagCategory::General);

        Ok(TaggingResult::new(&rating, &character, &general))
    }

    /// Predict the tags of a batch of images.
    pub fn predict_batch(
        &self,
        images: Vec<DynamicImage>,
    ) -> Result<Vec<TaggingResult>, TaggerError> {
        let tensor = self.preprocessor.process_batch(images)?;
        let probs = self.model.predict(tensor)?;
        let pairs = self.tags.create_probality_pairs(probs)?;

        let results = pairs
            .iter()
            .map(|pairs| {
                macro_rules! filter_tags {
                    ($category:expr) => {
                        pairs
                            .iter()
                            .filter(|(tag, &prob)| {
                                self.tags.label2tag().get(tag.clone()).unwrap().category()
                                    == $category
                                    && &prob >= &self.threshold
                            })
                            .map(|(tag, prob)| (tag.clone(), *prob))
                            .collect::<Prediction>()
                    };
                }
                let rating: Prediction = filter_tags!(TagCategory::Rating);
                let character: Prediction = filter_tags!(TagCategory::Character);
                let general: Prediction = filter_tags!(TagCategory::General);

                TaggingResult::new(&rating, &character, &general)
            })
            .collect::<Vec<TaggingResult>>();

        Ok(results)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pipe_from_pretrained() {
        let pipeline =
            TaggingPipeline::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3", Device::cpu())
                .unwrap();
        dbg!(&pipeline);
    }

    #[test]
    fn test_tagging_pipeline() {
        let pipeline =
            TaggingPipeline::from_pretrained("SmilingWolf/wd-swinv2-tagger-v3", Device::cpu())
                .unwrap();
        let image = image::open("assets/sample1_3x1024x1024.webp").unwrap();
        let result = pipeline.predict(image).unwrap();

        dbg!("Rating:", &result.rating);

        // get top 10 descending pairs
        let mut sorted = result.general.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        let top10 = sorted
            .iter()
            .take(10)
            .map(|(tag, prob)| (*tag, *prob))
            .collect::<Vec<_>>();
        dbg!("Top 10:", &top10);

        let top10keys = top10.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        assert_eq!(
            top10keys,
            vec![
                // https://huggingface.co/spaces/SmilingWolf/wd-tagger
                "1girl",
                "solo",
                "double_bun",
                "hair_bun",
                "twintails",
                "pink_hair",
                "fang",
                "smile",
                "pink_eyes",
                "looking_at_viewer",
            ]
        );

        // get last 10 tags
        let last10 = sorted
            .iter()
            .rev()
            .take(10)
            .map(|(tag, prob)| (tag, *prob))
            .collect::<IndexMap<_, _>>();
        dbg!("Last 10:", &last10);
    }
}
