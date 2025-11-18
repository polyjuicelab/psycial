//! High-level API for MBTI personality prediction
//!
//! This module provides a simple, user-friendly interface for loading models
//! and making predictions.
//!
//! # Quick Start
//!
//! ```no_run
//! use psycial::api::Predictor;
//!
//! let predictor = Predictor::new()?;
//! let result = predictor.predict("I love solving complex problems")?;
//! println!("{}", result.mbti_type); // e.g., "INTJ"
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Features
//!
//! ## Auto-Download from Hugging Face
//!
//! Enable the `auto-download` feature to automatically download models:
//!
//! ```toml
//! [dependencies]
//! psycial = { version = "0.1", features = ["auto-download"] }
//! ```
//!
//! ## Manual Model Download
//!
//! If not using auto-download, download models to `models/` directory:
//!
//! ```bash
//! mkdir -p models
//! wget https://huggingface.co/ElderRyan/psycial/resolve/main/mlp_weights_multitask.pt
//! wget https://huggingface.co/ElderRyan/psycial/resolve/main/tfidf_vectorizer_multitask.json
//! ```
//!
//! # Examples
//!
//! ## Batch Predictions
//!
//! ```no_run
//! # use psycial::api::Predictor;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictor = Predictor::new()?;
//! let texts = vec!["Text 1", "Text 2", "Text 3"];
//! let results = predictor.predict_batch(&texts)?;
//!
//! for (text, result) in texts.iter().zip(results.iter()) {
//!     println!("{}: {}", text, result.mbti_type);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Model Directory
//!
//! ```no_run
//! use psycial::api::{Predictor, PredictorConfig};
//!
//! let config = PredictorConfig::new()
//!     .with_model_dir("/custom/path")
//!     .with_auto_download(false);
//!
//! let predictor = Predictor::with_config(config)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Dimension-Level Predictions
//!
//! ```no_run
//! # use psycial::api::Predictor;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let predictor = Predictor::new()?;
//! let result = predictor.predict("Your text")?;
//!
//! println!("E/I: {} ({:.1}%)", result.dimensions.e_i.letter, result.dimensions.e_i.confidence * 100.0);
//! println!("S/N: {} ({:.1}%)", result.dimensions.s_n.letter, result.dimensions.s_n.confidence * 100.0);
//! println!("T/F: {} ({:.1}%)", result.dimensions.t_f.letter, result.dimensions.t_f.confidence * 100.0);
//! println!("J/P: {} ({:.1}%)", result.dimensions.j_p.letter, result.dimensions.j_p.confidence * 100.0);
//! # Ok(())
//! # }
//! ```
//!
//! ## Error Handling
//!
//! ```no_run
//! use psycial::api::Predictor;
//!
//! match Predictor::new() {
//!     Ok(predictor) => {
//!         // Use predictor
//!     }
//!     Err(e) => {
//!         eprintln!("Failed to load model: {}", e);
//!         eprintln!("Try: cargo build --features auto-download");
//!     }
//! }
//! ```

use crate::bert_embedder::BertEmbedder;
use crate::model_loader::ensure_model_files;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::TfidfVectorizer;

pub use crate::model_loader::ModelLoaderConfig as PredictorConfig;

/// Main predictor interface for MBTI personality classification
pub struct Predictor {
    model: MultiTaskGpuMLP,
    vectorizer: TfidfVectorizer,
    embedder: BertEmbedder,
    device: tch::Device,
}

/// Prediction result containing MBTI type and confidence scores
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// The predicted MBTI type (e.g., "INTJ")
    pub mbti_type: String,

    /// Overall confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Individual dimension predictions
    pub dimensions: DimensionPredictions,
}

/// Individual MBTI dimension predictions with confidence
#[derive(Debug, Clone)]
pub struct DimensionPredictions {
    pub e_i: DimensionResult,
    pub s_n: DimensionResult,
    pub t_f: DimensionResult,
    pub j_p: DimensionResult,
}

#[derive(Debug, Clone)]
pub struct DimensionResult {
    /// The predicted letter (e.g., "I", "N", "T", "J")
    pub letter: char,

    /// Confidence for this dimension (0.0 to 1.0)
    pub confidence: f64,
}

impl Predictor {
    /// Create a new predictor with default configuration
    ///
    /// This will:
    /// - Use the default model directory ("models/")
    /// - Load the multi-task model
    /// - Auto-download from HuggingFace if files don't exist (when feature enabled)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model files are not found and auto-download is disabled
    /// - Model files are corrupted
    /// - GPU/CUDA is not available (when GPU feature is enabled)
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_config(PredictorConfig::default())
    }

    /// Create a new predictor with custom configuration
    pub fn with_config(config: PredictorConfig) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure model files exist (download if needed)
        let files = config.get_model_files();
        ensure_model_files(&files, config.auto_download)?;

        // Determine device
        let device = if tch::Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        };

        // Load vectorizer
        let vectorizer =
            TfidfVectorizer::load(files.vectorizer.to_str().ok_or("Invalid vectorizer path")?)?;

        // Load BERT embedder
        let embedder = BertEmbedder::new()?;

        // Load model (using default architecture: 5384 input, [1024, 512, 256] hidden, 0.5 dropout)
        // TF-IDF: 5000 dims, BERT: 384 dims = 5384 total
        let model = MultiTaskGpuMLP::load(
            files.weights.to_str().ok_or("Invalid weights path")?,
            5384,
            vec![1024, 512, 256],
            0.5,
        )?;

        Ok(Self {
            model,
            vectorizer,
            embedder,
            device,
        })
    }

    /// Predict MBTI personality type from text
    ///
    /// # Arguments
    ///
    /// * `text` - The input text to analyze (can be a sentence, paragraph, or longer text)
    ///
    /// # Returns
    ///
    /// A `PredictionResult` containing the predicted MBTI type and confidence scores
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use psycial::api::Predictor;
    /// let predictor = Predictor::new().unwrap();
    /// let result = predictor.predict("I enjoy analyzing complex systems").unwrap();
    /// assert_eq!(result.mbti_type.len(), 4); // Always 4 letters
    /// ```
    pub fn predict(&self, text: &str) -> Result<PredictionResult, Box<dyn std::error::Error>> {
        // Extract features
        let tfidf_features = self.vectorizer.transform(text);
        let bert_features = self.embedder.embed(text)?;

        // Combine features
        let mut combined = tfidf_features;
        combined.extend(bert_features);

        // Get prediction with confidence
        let (mbti_type, confidences) = self.model.predict_with_confidence(&combined);

        // Parse MBTI type to get individual letters
        let chars: Vec<char> = mbti_type.chars().collect();
        if chars.len() != 4 {
            return Err(format!("Invalid MBTI type: {}", mbti_type).into());
        }

        let e_i = chars[0];
        let s_n = chars[1];
        let t_f = chars[2];
        let j_p = chars[3];

        // Calculate overall confidence (average of all dimensions)
        let confidence = (confidences[0] + confidences[1] + confidences[2] + confidences[3]) / 4.0;

        Ok(PredictionResult {
            mbti_type: mbti_type.clone(),
            confidence,
            dimensions: DimensionPredictions {
                e_i: DimensionResult {
                    letter: e_i,
                    confidence: confidences[0],
                },
                s_n: DimensionResult {
                    letter: s_n,
                    confidence: confidences[1],
                },
                t_f: DimensionResult {
                    letter: t_f,
                    confidence: confidences[2],
                },
                j_p: DimensionResult {
                    letter: j_p,
                    confidence: confidences[3],
                },
            },
        })
    }

    /// Predict MBTI types for multiple texts in batch
    ///
    /// More efficient than calling `predict` multiple times.
    pub fn predict_batch(
        &self,
        texts: &[&str],
    ) -> Result<Vec<PredictionResult>, Box<dyn std::error::Error>> {
        // Extract features for all texts
        let mut all_features = Vec::new();

        for text in texts {
            let tfidf_features = self.vectorizer.transform(text);
            let bert_features = self.embedder.embed(text)?;

            let mut combined = tfidf_features;
            combined.extend(bert_features);
            all_features.push(combined);
        }

        // Batch prediction
        let predictions = self.model.predict_batch_with_confidence(&all_features);

        // Convert to results
        predictions
            .iter()
            .map(|(mbti_type, confidences)| {
                let chars: Vec<char> = mbti_type.chars().collect();
                if chars.len() != 4 {
                    return Err(format!("Invalid MBTI type: {}", mbti_type).into());
                }

                let e_i = chars[0];
                let s_n = chars[1];
                let t_f = chars[2];
                let j_p = chars[3];

                let confidence =
                    (confidences[0] + confidences[1] + confidences[2] + confidences[3]) / 4.0;

                Ok(PredictionResult {
                    mbti_type: mbti_type.clone(),
                    confidence,
                    dimensions: DimensionPredictions {
                        e_i: DimensionResult {
                            letter: e_i,
                            confidence: confidences[0],
                        },
                        s_n: DimensionResult {
                            letter: s_n,
                            confidence: confidences[1],
                        },
                        t_f: DimensionResult {
                            letter: t_f,
                            confidence: confidences[2],
                        },
                        j_p: DimensionResult {
                            letter: j_p,
                            confidence: confidences[3],
                        },
                    },
                })
            })
            .collect()
    }

    /// Get information about the loaded model
    pub fn model_info(&self) -> ModelInfo {
        ModelInfo {
            device: format!("{:?}", self.device),
            input_dim: self.vectorizer.vocabulary.len() + 384, // TF-IDF + BERT
        }
    }
}

/// Information about the loaded model
#[derive(Debug)]
pub struct ModelInfo {
    pub device: String,
    pub input_dim: usize,
}

impl std::fmt::Display for PredictionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MBTI Type: {}", self.mbti_type)?;
        writeln!(f, "Overall Confidence: {:.2}%", self.confidence * 100.0)?;
        writeln!(f, "\nDimension Breakdown:")?;
        writeln!(
            f,
            "  E/I: {} ({:.2}%)",
            self.dimensions.e_i.letter,
            self.dimensions.e_i.confidence * 100.0
        )?;
        writeln!(
            f,
            "  S/N: {} ({:.2}%)",
            self.dimensions.s_n.letter,
            self.dimensions.s_n.confidence * 100.0
        )?;
        writeln!(
            f,
            "  T/F: {} ({:.2}%)",
            self.dimensions.t_f.letter,
            self.dimensions.t_f.confidence * 100.0
        )?;
        writeln!(
            f,
            "  J/P: {} ({:.2}%)",
            self.dimensions.j_p.letter,
            self.dimensions.j_p.confidence * 100.0
        )?;
        Ok(())
    }
}
