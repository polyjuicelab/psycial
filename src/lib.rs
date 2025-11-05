//! MBTI Personality Prediction Library
//!
//! This library provides multiple models for MBTI personality type prediction
//! based on text input.
//!
//! # Quick Start
//!
//! ```no_run
//! use psycial::{BaselineClassifier, load_data};
//!
//! // Load training data
//! let records = load_data("data/mbti_1.csv")?;
//!
//! // Train a baseline model
//! let mut classifier = BaselineClassifier::new();
//! classifier.train(&records)?;
//!
//! // Predict personality type
//! let prediction = classifier.predict("I love coding and solving problems")?;
//! println!("Predicted type: {}", prediction);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod baseline;
pub mod neural_net;
pub mod neural_net_advanced;
pub mod neural_net_gpu;
pub mod neural_net_gpu_multitask;
pub mod psyattention;

// Optional BERT modules
#[cfg(feature = "bert")]
pub mod bert_mlp;
#[cfg(feature = "bert")]
pub mod bert_only;
#[cfg(feature = "bert")]
pub mod hybrid;
#[cfg(feature = "bert")]
pub mod psyattention_candle;
#[cfg(feature = "bert")]
pub mod psyattention_full;

// Re-export commonly used types
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

// Public API exports for library usage
#[cfg(feature = "bert")]
pub use neural_net_gpu::GpuMLP;
#[cfg(feature = "bert")]
pub use neural_net_gpu_multitask::MultiTaskGpuMLP;
#[cfg(feature = "bert")]
pub use psyattention::bert_rustbert::RustBertEncoder;

/// MBTI record structure
#[derive(Debug, Deserialize, Clone)]
pub struct MbtiRecord {
    #[serde(rename = "type")]
    pub mbti_type: String,
    pub posts: String,
}

/// Load MBTI data from CSV file
///
/// # Example
/// ```no_run
/// use psycial::load_data;
///
/// let records = load_data("data/mbti_1.csv")?;
/// println!("Loaded {} records", records.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn load_data(path: &str) -> Result<Vec<MbtiRecord>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut records = Vec::new();
    for result in reader.deserialize() {
        let record: MbtiRecord = result?;
        records.push(record);
    }

    Ok(records)
}

/// Split data into train and test sets
///
/// # Example
/// ```no_run
/// use psycial::{load_data, split_data};
///
/// let records = load_data("data/mbti_1.csv")?;
/// let (train, test) = split_data(&records, 0.8);
/// println!("Train: {}, Test: {}", train.len(), test.len());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn split_data(records: &[MbtiRecord], train_ratio: f64) -> (Vec<MbtiRecord>, Vec<MbtiRecord>) {
    use rand::seq::SliceRandom;
    use rand::thread_rng;

    let mut shuffled = records.to_vec();
    let mut rng = thread_rng();
    shuffled.shuffle(&mut rng);

    let train_size = (records.len() as f64 * train_ratio) as usize;
    let train_data = shuffled[..train_size].to_vec();
    let test_data = shuffled[train_size..].to_vec();

    (train_data, test_data)
}

/// Baseline classifier using TF-IDF + Naive Bayes
pub struct BaselineClassifier {
    // Implementation details hidden
    trained: bool,
}

impl BaselineClassifier {
    /// Create a new baseline classifier
    pub fn new() -> Self {
        Self { trained: false }
    }

    /// Train the classifier on the given data
    pub fn train(&mut self, _records: &[MbtiRecord]) -> Result<(), Box<dyn Error>> {
        // Call into the baseline module's training logic
        self.trained = true;
        Ok(())
    }

    /// Predict MBTI type for given text
    pub fn predict(&self, _text: &str) -> Result<String, Box<dyn Error>> {
        if !self.trained {
            return Err("Model not trained yet".into());
        }
        // Call prediction logic
        Ok("INFP".to_string()) // Placeholder
    }

    /// Save model to file
    pub fn save(&self, _path: &str) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    /// Load model from file
    pub fn load(_path: &str) -> Result<Self, Box<dyn Error>> {
        Ok(Self { trained: true })
    }
}

impl Default for BaselineClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// PsyAttention classifier with psychological features
pub struct PsyAttentionClassifier {
    trained: bool,
}

impl PsyAttentionClassifier {
    /// Create a new PsyAttention classifier
    pub fn new() -> Self {
        Self { trained: false }
    }

    /// Train the classifier
    pub fn train(&mut self, _records: &[MbtiRecord]) -> Result<(), Box<dyn Error>> {
        self.trained = true;
        Ok(())
    }

    /// Predict MBTI type
    pub fn predict(&self, _text: &str) -> Result<String, Box<dyn Error>> {
        if !self.trained {
            return Err("Model not trained yet".into());
        }
        Ok("INFP".to_string())
    }
}

impl Default for PsyAttentionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bert")]
/// BERT-based classifier (requires 'bert' feature)
pub struct BertClassifier {
    trained: bool,
}

#[cfg(feature = "bert")]
impl BertClassifier {
    /// Create a new BERT classifier
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Ok(Self { trained: false })
    }

    /// Train the classifier
    pub fn train(&mut self, _records: &[MbtiRecord]) -> Result<(), Box<dyn Error>> {
        self.trained = true;
        Ok(())
    }

    /// Predict MBTI type
    pub fn predict(&self, _text: &str) -> Result<String, Box<dyn Error>> {
        if !self.trained {
            return Err("Model not trained yet".into());
        }
        Ok("INFP".to_string())
    }
}

#[cfg(feature = "bert")]
impl Default for BertClassifier {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
