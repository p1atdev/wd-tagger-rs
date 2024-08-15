use crate::error::TaggerError;
use anyhow::Result;
use hf_hub::{api::sync::ApiBuilder, Cache, Repo, RepoType};
use std::path::PathBuf;

/// Trait for the HuggingFace file
pub trait HfFile {
    /// Download or use cache using default cache config and return the file path
    fn get(&self) -> Result<PathBuf, TaggerError> {
        self.get_with_cache(Cache::default())
    }

    /// Download or use cache using specified cache config and return the file path
    fn get_with_cache(&self, cache: Cache) -> Result<PathBuf, TaggerError>;
}

/// Model for the Tagging
pub struct TaggerModelFile {
    repo_id: String,
    revision: Option<String>,
    model_path: String,
}

impl TaggerModelFile {
    /// Initialize simply with the repo_id
    pub fn new(repo_id: String) -> Self {
        Self {
            repo_id,
            revision: None,
            model_path: "model.onnx".to_string(),
        }
    }

    /// Initialize with the repo_id, revision, and model_path
    pub fn custom(repo_id: String, revision: Option<String>, model_path: String) -> Self {
        Self {
            repo_id,
            revision,
            model_path,
        }
    }

    /// Get the repo_id
    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    /// Get the revision
    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
    }

    /// Get the model_path
    pub fn model_path(&self) -> &str {
        &self.model_path
    }
}

impl HfFile for TaggerModelFile {
    fn get_with_cache(&self, cache: Cache) -> Result<PathBuf, TaggerError> {
        let builder = ApiBuilder::from_cache(cache);
        let api = match builder.build() {
            Ok(api) => api,
            Err(e) => return Err(TaggerError::Hf(format!("Error while building API: {}", e))),
        };

        let repo = match &self.revision {
            Some(revision) => api.repo(Repo::with_revision(
                self.repo_id.clone(),
                RepoType::Model,
                revision.clone(),
            )),
            None => api.repo(Repo::new(self.repo_id.clone(), RepoType::Model)),
        };

        match repo.get(&self.model_path) {
            Ok(path) => Ok(path),
            Err(e) => Err(TaggerError::Hf(format!("Error getting model: {}", e))),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_get_model() {
        let repo_id = "SmilingWolf/wd-swinv2-tagger-v3".to_string();
        let revision = "main".to_string();
        let model_path = "model.onnx".to_string();

        let model_file = TaggerModelFile::new(repo_id.clone());

        assert_eq!(&model_file.repo_id(), &repo_id);
        assert_eq!(&model_file.revision(), &None);
        assert_eq!(&model_file.model_path(), &model_path);

        let path = model_file.get().unwrap();

        assert!(path.exists());

        let model_file_custom = TaggerModelFile::custom(repo_id, Some(revision), model_path);

        assert!(model_file_custom.get().is_ok());
    }
}
