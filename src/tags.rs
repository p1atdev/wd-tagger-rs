use std::{collections::HashMap, fs::File, path::Path};

use anyhow::Result;
use serde::Deserialize;

use crate::error::TaggerError;

/// Each record in the CSV file
#[derive(Debug, Deserialize)]
struct Tag {
    tag_id: i32,
    name: String,
    category: TagCategory,
    count: i32,
}

/// Tag category
#[derive(Debug, Deserialize)]
pub enum TagCategory {
    #[serde(rename = "0")]
    General,
    #[serde(rename = "1")]
    Artist,
    #[serde(rename = "3")]
    Copyright,
    #[serde(rename = "4")]
    Character,
    #[serde(rename = "5")]
    Meta,
    #[serde(rename = "9")]
    Rating,
}

/// The tags in the CSV file
pub struct LabelTags {
    tags: Vec<Tag>,
}

impl LabelTags {
    /// Load from the local CSV file
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, TaggerError> {
        let file = File::open(csv_path).map_err(|e| TaggerError::Tag(e.to_string()))?;
        let mut rdr = csv::Reader::from_reader(file);

        let tags = rdr
            .deserialize()
            .map(|result| result.map_err(|e| TaggerError::Tag(e.to_string())))
            .collect::<Result<Vec<Tag>, TaggerError>>()?;

        Ok(Self { tags })
    }

    /// Create pairs of tag and probability with given tensor
    pub fn create_probality_pairs(
        &self,
        tensor: Vec<Vec<f32>>,
    ) -> Result<Vec<HashMap<String, f32>>, TaggerError> {
        fn map_pair(
            tags: &Vec<Tag>,
            probs: &Vec<f32>,
        ) -> Result<HashMap<String, f32>, TaggerError> {
            if &tags.len() != &probs.len() {
                return Err(TaggerError::Tag(
                    "Tags and probabilities length mismatch".to_string(),
                ));
            }

            Ok(probs
                .iter()
                .zip(tags)
                .map(|(prob, tag)| (tag.name.clone(), *prob))
                .collect::<HashMap<String, f32>>())
        }

        tensor
            .iter() // batch
            .map(|probs| map_pair(&self.tags, &probs))
            .collect::<Result<Vec<HashMap<String, f32>>, TaggerError>>()
    }
}

#[cfg(test)]
mod test {
    use rand::random;

    use super::*;

    use crate::file::{HfFile, TagCSVFile};

    #[test]
    fn test_load_tags() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3".to_string())
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        dbg!(&tags.tags[0..5]);
    }

    #[test]
    fn test_create_probs_valid() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3".to_string())
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        let random_prob = (0..4)
            .map(|_| {
                (0..tags.tags.len())
                    .map(|_| random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let _pairs = tags.create_probality_pairs(random_prob).unwrap();
    }

    #[test]
    fn test_create_probs_valid_invalid() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3".to_string())
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        let random_prob = (0..4)
            .map(|_| {
                (0..tags.tags.len() + 100) // wrong size!
                    .map(|_| random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let pairs = tags.create_probality_pairs(random_prob);
        assert!(pairs.is_err());
    }
}
