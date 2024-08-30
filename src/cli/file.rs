use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::path::PathBuf;
use tokio::fs;
use tokio::fs::{File, ReadDir};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Supported image extensions.
pub const IMAGE_EXTENSIONS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];

/// Check if the path is a file or directory.
pub async fn is_file(path: &str) -> Result<bool> {
    let metadata = fs::metadata(path).await?;
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
pub async fn write_text_to_file(text: &str, path: &str) -> Result<()> {
    let mut file = File::create(path).await?;
    file.write_all(text.as_bytes()).await?;
    Ok(())
}

/// Create a directory.
pub async fn create_dir(path: &str) -> Result<()> {
    fs::create_dir(path).await?;
    Ok(())
}
