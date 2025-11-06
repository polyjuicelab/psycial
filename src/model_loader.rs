//! Model loading utilities with optional auto-download from Hugging Face
//!
//! This module handles model file management, including automatic downloading
//! from Hugging Face when the `auto-download` feature is enabled.
//!
//! # Examples
//!
//! ## Check if models exist
//!
//! ```no_run
//! use psycial::model_loader::ModelFiles;
//!
//! let files = ModelFiles::multitask(None);
//! if files.exists() {
//!     println!("Models are ready!");
//! }
//! ```
//!
//! ## Manual download (with auto-download feature)
//!
//! ```no_run
//! # #[cfg(feature = "auto-download")]
//! # {
//! use psycial::model_loader::{ModelFiles, ensure_model_files};
//!
//! let files = ModelFiles::multitask(None);
//! ensure_model_files(&files, true)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }
//! ```
//!
//! ## Custom model directory
//!
//! ```no_run
//! use psycial::model_loader::{ModelLoaderConfig, ModelType};
//!
//! let config = ModelLoaderConfig::new()
//!     .with_model_dir("/custom/models")
//!     .with_model_type(ModelType::SingleTask);
//!
//! let files = config.get_model_files();
//! ```

use std::path::{Path, PathBuf};
use std::fs;

const HF_REPO_ID: &str = "ElderRyan/psycial";
const DEFAULT_MODEL_DIR: &str = "models";

/// Model files that need to be downloaded
pub struct ModelFiles {
    pub weights: PathBuf,
    pub vectorizer: PathBuf,
}

impl ModelFiles {
    /// Get paths for multi-task model files
    pub fn multitask(model_dir: Option<&str>) -> Self {
        let base = model_dir.unwrap_or(DEFAULT_MODEL_DIR);
        Self {
            weights: PathBuf::from(base).join("mlp_weights_multitask.pt"),
            vectorizer: PathBuf::from(base).join("tfidf_vectorizer_multitask.json"),
        }
    }

    /// Get paths for single-task model files
    pub fn single_task(model_dir: Option<&str>) -> Self {
        let base = model_dir.unwrap_or(DEFAULT_MODEL_DIR);
        Self {
            weights: PathBuf::from(base).join("mlp_weights_single.pt"),
            vectorizer: PathBuf::from(base).join("tfidf_vectorizer_single.json"),
        }
    }

    /// Check if all required files exist
    pub fn exists(&self) -> bool {
        self.weights.exists() && self.vectorizer.exists()
    }
}

/// Load model files, optionally downloading from Hugging Face if not found locally
///
/// # Arguments
/// * `files` - Model file paths to load
/// * `auto_download` - Whether to automatically download from HF if files don't exist
///
/// # Example
/// ```no_run
/// use psycial::model_loader::{ModelFiles, ensure_model_files};
///
/// let files = ModelFiles::multitask(None);
/// ensure_model_files(&files, true).expect("Failed to load model files");
/// ```
pub fn ensure_model_files(files: &ModelFiles, auto_download: bool) -> Result<(), Box<dyn std::error::Error>> {
    if files.exists() {
        return Ok(());
    }

    if !auto_download {
        return Err(format!(
            "Model files not found. Expected:\n  - {}\n  - {}\n\nEither:\n1. Train the model first\n2. Enable 'auto-download' feature and rebuild",
            files.weights.display(),
            files.vectorizer.display()
        ).into());
    }

    #[cfg(feature = "auto-download")]
    {
        println!("Model files not found locally. Downloading from Hugging Face...");
        download_from_hf(files)?;
        println!("✓ Model files downloaded successfully");
        Ok(())
    }

    #[cfg(not(feature = "auto-download"))]
    {
        Err("auto-download feature not enabled. Rebuild with --features auto-download".into())
    }
}

#[cfg(feature = "auto-download")]
fn download_from_hf(files: &ModelFiles) -> Result<(), Box<dyn std::error::Error>> {
    use hf_hub::api::sync::Api;

    // Ensure models directory exists
    if let Some(parent) = files.weights.parent() {
        fs::create_dir_all(parent)?;
    }

    let api = Api::new()?;
    let repo = api.model(HF_REPO_ID.to_string());

    // Download weights
    let weights_filename = files.weights.file_name()
        .and_then(|s| s.to_str())
        .ok_or("Invalid weights filename")?;
    println!("  Downloading {}...", weights_filename);
    let downloaded_weights = repo.get(weights_filename)?;
    fs::copy(&downloaded_weights, &files.weights)?;

    // Download vectorizer
    let vec_filename = files.vectorizer.file_name()
        .and_then(|s| s.to_str())
        .ok_or("Invalid vectorizer filename")?;
    println!("  Downloading {}...", vec_filename);
    let downloaded_vec = repo.get(vec_filename)?;
    fs::copy(&downloaded_vec, &files.vectorizer)?;

    Ok(())
}

/// Configuration for model loading
pub struct ModelLoaderConfig {
    pub model_dir: Option<String>,
    pub auto_download: bool,
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    MultiTask,
    SingleTask,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            model_dir: None,
            auto_download: cfg!(feature = "auto-download"),
            model_type: ModelType::MultiTask,
        }
    }
}

impl ModelLoaderConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_dir(mut self, dir: impl Into<String>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    pub fn with_auto_download(mut self, enable: bool) -> Self {
        self.auto_download = enable;
        self
    }

    pub fn with_model_type(mut self, model_type: ModelType) -> Self {
        self.model_type = model_type;
        self
    }

    pub fn get_model_files(&self) -> ModelFiles {
        match self.model_type {
            ModelType::MultiTask => ModelFiles::multitask(self.model_dir.as_deref()),
            ModelType::SingleTask => ModelFiles::single_task(self.model_dir.as_deref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_files_paths() {
        let files = ModelFiles::multitask(None);
        assert_eq!(files.weights, PathBuf::from("models/mlp_weights_multitask.pt"));
        assert_eq!(files.vectorizer, PathBuf::from("models/tfidf_vectorizer_multitask.json"));
    }

    #[test]
    fn test_custom_model_dir() {
        let files = ModelFiles::multitask(Some("/tmp/models"));
        assert_eq!(files.weights, PathBuf::from("/tmp/models/mlp_weights_multitask.pt"));
    }

    #[test]
    fn test_config_builder() {
        let config = ModelLoaderConfig::new()
            .with_model_dir("custom_models")
            .with_auto_download(true);
        
        assert_eq!(config.model_dir, Some("custom_models".to_string()));
        assert!(config.auto_download);
    }
}

