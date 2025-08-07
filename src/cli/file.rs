use anyhow::Result;
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use wdtagger::pipeline::TaggingResult;

use crate::tag::fix_tag_underscore;

/// Supported image extensions.
pub const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];

/// Check if the path is a file or directory.
pub async fn is_file<P: AsRef<Path>>(path: P) -> Result<bool> {
    let metadata = fs::metadata(&path).await?;
    Ok(metadata.is_file())
}

/// Check if the path is an image file.
pub fn is_image(path: &str) -> Result<bool> {
    match PathBuf::from(path).extension() {
        Some(ext) => {
            let ext = ext.to_string_lossy().to_lowercase();
            Ok(IMAGE_EXTENSIONS.contains(&ext.as_str()))
        }
        None => Ok(false),
    }
}

/// Get image files from a directory.
pub async fn get_image_files(dir: &str) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir).await?;
    let mut tasks = vec![];

    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        let task = tokio::spawn(async move {
            if is_image(path.to_str().unwrap()).unwrap() {
                Some(path)
            } else {
                None
            }
        });

        tasks.push(task);
    }

    let files = stream::iter(tasks)
        .buffer_unordered(16)
        .filter_map(|result| async move {
            match result {
                Ok(Some(path)) => Some(path),
                _ => None,
            }
        })
        .collect()
        .await;

    Ok(files)
}

/// Write a text to a file.
pub async fn write_text_to_file(text: &str, path: &PathBuf) -> Result<()> {
    let mut file = File::create(path).await?;
    file.write_all(text.as_bytes()).await?;
    Ok(())
}

/// Create a directory.
pub async fn create_dir(path: &str) -> Result<()> {
    fs::create_dir(path).await?;
    Ok(())
}

pub fn get_path_with_extension<P: AsRef<Path>>(path: P, ext: &str) -> PathBuf {
    let mut new_path = path.as_ref().to_path_buf();
    new_path.set_extension(ext);
    new_path
}

#[derive(Serialize, Deserialize)]
pub struct TaggingResultDetail {
    rating: HashMap<String, f32>,
    character: HashMap<String, f32>,
    general: HashMap<String, f32>,
}

impl From<TaggingResult> for TaggingResultDetail {
    fn from(result: TaggingResult) -> Self {
        Self {
            rating: result.rating.into_iter().collect::<HashMap<_, _>>(),
            character: result.character.into_iter().collect::<HashMap<_, _>>(),
            general: result.general.into_iter().collect::<HashMap<_, _>>(),
        }
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct TaggingResultSimpleTags {
    pub rating: String,
    pub character: Vec<String>,
    pub general: Vec<String>,
}

#[derive(Serialize, Debug, Clone)]
pub struct TaggingResultSimple {
    pub tags: String,
    pub tagger: TaggingResultSimpleTags,
}

impl From<TaggingResult> for TaggingResultSimpleTags {
    fn from(result: TaggingResult) -> Self {
        Self {
            rating: result
                .rating
                .first()
                .map_or("".to_string(), |(k, _)| k.clone()),
            character: result
                .character
                .keys()
                .map(|tag| fix_tag_underscore(&tag))
                .collect(),
            general: result
                .general
                .keys()
                .map(|tag| fix_tag_underscore(&tag))
                .collect(),
        }
    }
}

impl From<TaggingResult> for TaggingResultSimple {
    fn from(result: TaggingResult) -> Self {
        let mut tags = result.character.keys().cloned().collect::<Vec<String>>();
        tags.extend(result.general.keys().cloned().collect::<Vec<String>>());

        let tags = tags
            .iter()
            .map(|tag| fix_tag_underscore(tag))
            .collect::<Vec<String>>()
            .join(", ");

        Self {
            tags,
            tagger: TaggingResultSimpleTags::from(result),
        }
    }
}

pub async fn write_as_json<T: Serialize>(path: &PathBuf, result: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(&result)?;
    write_text_to_file(&json, path).await
}

pub async fn write_as_caption(path: &PathBuf, result: &TaggingResult) -> Result<()> {
    // let rating_tag = result
    //     .rating
    //     .first()
    //     .map(|(k, _)| k.clone())
    //     .unwrap_or("".to_string());
    let character_tags = result
        .character
        .keys()
        .map(|tag| fix_tag_underscore(&tag))
        .collect::<Vec<_>>()
        .join(", ");
    let general_tags = result
        .general
        .keys()
        .map(|tag| fix_tag_underscore(&tag))
        .collect::<Vec<_>>()
        .join(", ");

    let caption = vec![character_tags, general_tags]
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");

    write_text_to_file(&caption, path).await
}
