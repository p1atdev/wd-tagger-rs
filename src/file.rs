use crate::error::TaggerError;
use anyhow::Result;
use hf_hub::{
    api::sync::{Api, ApiBuilder, ApiRepo},
    Cache, Repo, RepoType,
};
use std::path::PathBuf;

/// Trait for the HuggingFace file
pub trait HfFile {
    /// Initialize simply with the repo_id
    fn new(repo_id: String) -> Self;

    /// Get the repo_id
    fn repo_id(&self) -> String;

    /// Get the revision
    fn revision(&self) -> Option<String>;

    /// Get the model_path
    fn file_path(&self) -> String;

    /// Get repo object
    fn _api_repo(&self, api: Api) -> ApiRepo {
        match self.revision() {
            Some(revision) => api.repo(Repo::with_revision(
                self.repo_id(),
                RepoType::Model,
                revision,
            )),
            None => api.repo(Repo::new(self.repo_id(), RepoType::Model)),
        }
    }

    /// Get file from the repo
    fn _get_file(&self, repo: ApiRepo, file_path: &str) -> Result<PathBuf, TaggerError> {
        match repo.get(&file_path) {
            Ok(path) => Ok(path),
            Err(e) => Err(TaggerError::Hf(format!("Error getting model: {}", e))),
        }
    }

    /// Download or use cache using default cache config and return the file path
    fn get(&self) -> Result<PathBuf, TaggerError> {
        self.get_with_cache(Cache::default())
    }

    /// Download or use cache using specified cache config and return the file path
    fn get_with_cache(&self, cache: Cache) -> Result<PathBuf, TaggerError> {
        let builder = ApiBuilder::from_cache(cache);
        let api = match builder.build() {
            Ok(api) => api,
            Err(e) => return Err(TaggerError::Hf(format!("Error while building API: {}", e))),
        };

        let repo = self._api_repo(api);

        self._get_file(repo, &self.file_path())
    }
}

/// Model for the Tagging
pub struct TaggerModelFile {
    repo_id: String,
    revision: Option<String>,
    model_path: String,
}

impl TaggerModelFile {
    /// Initialize with the repo_id, revision, and model_path
    pub fn custom(repo_id: String, revision: Option<String>, model_path: String) -> Self {
        Self {
            repo_id,
            revision,
            model_path,
        }
    }
}

impl HfFile for TaggerModelFile {
    fn new(repo_id: String) -> Self {
        Self {
            repo_id,
            revision: None,
            model_path: "model.onnx".to_string(),
        }
    }

    fn repo_id(&self) -> String {
        self.repo_id.clone()
    }

    fn revision(&self) -> Option<String> {
        self.revision.clone()
    }

    fn file_path(&self) -> String {
        self.model_path.clone()
    }
}

pub struct TagCSVFile {
    repo_id: String,
    revision: Option<String>,
    csv_path: String,
}

impl TagCSVFile {
    pub fn custom(repo_id: String, revision: Option<String>, csv_path: String) -> Self {
        Self {
            repo_id,
            revision,
            csv_path,
        }
    }
}

impl HfFile for TagCSVFile {
    fn new(repo_id: String) -> Self {
        Self {
            repo_id,
            revision: None,
            csv_path: "selected_tags.csv".to_string(),
        }
    }

    fn repo_id(&self) -> String {
        self.repo_id.clone()
    }

    fn revision(&self) -> Option<String> {
        self.revision.clone()
    }

    fn file_path(&self) -> String {
        self.csv_path.clone()
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
        assert_eq!(&model_file.file_path(), &model_path);

        let path = model_file.get().unwrap();

        assert!(path.exists());

        let model_file_custom = TaggerModelFile::custom(repo_id, Some(revision), model_path);

        assert!(model_file_custom.get().is_ok());
    }
}
