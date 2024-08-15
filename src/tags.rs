use std::collections::HashMap;
use std::{collections::BTreeMap, fs::File, path::Path};

use anyhow::Result;
use serde::Deserialize;

use crate::error::TaggerError;
use crate::file::{HfFile, TagCSVFile};

/// Each record in the CSV file
#[derive(Debug, Deserialize, Clone)]
pub struct Tag {
    tag_id: i32,
    name: String,
    category: TagCategory,
    count: i32,
}

/// Tag category
#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
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

impl Tag {
    pub fn category(&self) -> TagCategory {
        self.category.clone()
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn tag_id(&self) -> i32 {
        self.tag_id
    }

    pub fn count(&self) -> i32 {
        self.count
    }
}

/// The tags in the CSV file
#[derive(Debug, Clone)]
pub struct LabelTags {
    total_tags: usize,
    label2tag: HashMap<String, Tag>,
    idx2tag: HashMap<usize, Tag>,
}

impl LabelTags {
    /// Load from the local CSV file
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, TaggerError> {
        let file = File::open(csv_path).map_err(|e| TaggerError::Tag(e.to_string()))?;
        let mut rdr = csv::Reader::from_reader(file);

        let tag_list = rdr
            .deserialize()
            .map(|result| result.map_err(|e| TaggerError::Tag(e.to_string())))
            .collect::<Result<Vec<Tag>, TaggerError>>()?;

        let total_tags = tag_list.len();
        let label2tag = tag_list
            .iter()
            .map(|tag| (tag.name.clone(), tag.clone()))
            .collect::<HashMap<String, Tag>>();
        let idx2tag = tag_list
            .into_iter()
            .enumerate()
            .collect::<HashMap<usize, Tag>>();

        Ok(Self {
            total_tags,
            label2tag,
            idx2tag,
        })
    }

    pub fn from_pretrained(repo_id: &str) -> Result<Self, TaggerError> {
        let csv_path = TagCSVFile::new(repo_id).get()?;
        Self::load(csv_path)
    }

    /// Create pairs of tag and probability with given tensor
    pub fn create_probality_pairs(
        &self,
        tensor: Vec<Vec<f32>>,
    ) -> Result<Vec<HashMap<String, f32>>, TaggerError> {
        fn map_pair(
            idx2tags: &HashMap<usize, Tag>,
            probs: &Vec<f32>,
        ) -> Result<HashMap<String, f32>, TaggerError> {
            if &idx2tags.len() != &probs.len() {
                return Err(TaggerError::Tag(
                    "Tags and probabilities length mismatch".to_string(),
                ));
            }

            Ok(probs
                .iter()
                .enumerate()
                .map(|(idx, prob)| (idx2tags.get(&idx).unwrap().name(), prob.clone()))
                .collect::<HashMap<String, f32>>())
        }

        tensor
            .iter() // batch
            .map(|probs| map_pair(&self.idx2tag, &probs))
            .collect::<Result<Vec<HashMap<String, f32>>, TaggerError>>()
    }

    pub fn label2tag(&self) -> &HashMap<String, Tag> {
        &self.label2tag
    }

    pub fn idx2tag(&self) -> &HashMap<usize, Tag> {
        &self.idx2tag
    }
}

#[cfg(test)]
mod test {
    use rand::random;

    use super::*;

    use crate::file::{HfFile, TagCSVFile};

    #[test]
    fn test_load_tags() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3")
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        dbg!(&tags.label2tag().iter().take(5));

        // sort by idx, smaller idx first
        let mut top = tags.idx2tag().iter().collect::<Vec<_>>();
        top.sort_by(|(idx1, _), (idx2, _)| idx1.cmp(idx2));

        assert_eq!(
            top.iter()
                .take(5)
                .map(|(_idx, tag)| tag.name.clone())
                .collect::<Vec<_>>(),
            vec!["general", "sensitive", "questionable", "explicit", "1girl",]
        );
    }

    #[test]
    fn test_create_probs_valid() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3")
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        let random_prob = (0..4)
            .map(|_| {
                (0..tags.total_tags)
                    .map(|_| random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let _pairs = tags.create_probality_pairs(random_prob).unwrap();
    }

    #[test]
    fn test_create_probs_valid_invalid() {
        let csv_path = TagCSVFile::new("SmilingWolf/wd-swinv2-tagger-v3")
            .get()
            .unwrap();
        let tags = LabelTags::load(csv_path).unwrap();

        let random_prob = (0..4)
            .map(|_| {
                (0..tags.total_tags + 100) // wrong size!
                    .map(|_| random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let pairs = tags.create_probality_pairs(random_prob);
        assert!(pairs.is_err());
    }
}
