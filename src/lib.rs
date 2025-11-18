//! # Psycial - MBTI Personality Classifier
//!
//! Production-grade MBTI (Myers-Briggs Type Indicator) personality classification
//! with GPU-accelerated multi-task neural networks in pure Rust.
//!
//! ## Features
//!
//! - **52.05% Accuracy**: 8.3x better than random baseline (6.25%)
//! - **Multi-Task Learning**: Independent classifiers for each MBTI dimension  
//! - **GPU Acceleration**: CUDA support for fast inference
//! - **Auto-Download**: Fetch pre-trained models from Hugging Face
//! - **Hybrid Features**: TF-IDF (5000 dims) + BERT embeddings (384 dims)
//!
//! ## Quick Start
//!
//! Add to your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! # With auto-download from Hugging Face (recommended)
//! psycial = { version = "0.1", features = ["auto-download"] }
//!
//! # Or minimal (requires pre-downloaded models)
//! psycial = { version = "0.1", default-features = false }
//! ```
//!
//! ### Basic Usage
//!
//! ```no_run
//! use psycial::api::Predictor;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Model will auto-download from HuggingFace if not found
//!     let predictor = Predictor::new()?;
//!     
//!     let result = predictor.predict("I love solving complex problems")?;
//!     
//!     println!("MBTI Type: {}", result.mbti_type);  // e.g., "INTJ"
//!     println!("Confidence: {:.1}%", result.confidence * 100.0);
//!     
//!     // Access individual dimensions
//!     println!("E/I: {} ({:.1}%)", 
//!              result.dimensions.e_i.letter,
//!              result.dimensions.e_i.confidence * 100.0);
//!     
//!     Ok(())
//! }
//! ```
//!
//! ### Batch Predictions
//!
//! ```no_run
//! # use psycial::api::Predictor;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let predictor = Predictor::new()?;
//!
//! let texts = vec![
//!     "I enjoy social gatherings",
//!     "I prefer working alone",
//! ];
//!
//! let results = predictor.predict_batch(&texts)?;
//! for (text, result) in texts.iter().zip(results) {
//!     println!("{}: {}", text, result.mbti_type);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom Configuration
//!
//! ```no_run
//! # use psycial::api::{Predictor, PredictorConfig};
//! # use psycial::model_loader::ModelType;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = PredictorConfig::new()
//!     .with_model_dir("custom_models")
//!     .with_model_type(ModelType::MultiTask)
//!     .with_auto_download(true);
//!
//! let predictor = Predictor::with_config(config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! ### Available Cargo Features
//!
//! | Feature | Description | Default |
//! |---------|-------------|---------|
//! | `cli` | Include CLI binary | ✓ |
//! | `auto-download` | Auto-download models from HuggingFace | ✗ |
//! | `gpu` | Enable GPU acceleration (requires CUDA) | ✗ |
//!
//! ### Feature Examples
//!
//! ```toml
//! # Minimal library only (no CLI, manual model download)
//! psycial = { version = "0.1", default-features = false }
//!
//! # Library with auto-download, no CLI
//! psycial = { version = "0.1", default-features = false, features = ["auto-download"] }
//!
//! # Full features including GPU
//! psycial = { version = "0.1", features = ["auto-download", "gpu"] }
//! ```
//!
//! ## Manual Model Download
//!
//! If you don't use the `auto-download` feature, download models manually:
//!
//! ### Using Python:
//!
//! ```python
//! from huggingface_hub import hf_hub_download
//! import shutil
//!
//! weights = hf_hub_download('ElderRyan/psycial', 'mlp_weights_multitask.pt')
//! vec = hf_hub_download('ElderRyan/psycial', 'tfidf_vectorizer_multitask.json')
//!
//! shutil.copy(weights, 'models/mlp_weights_multitask.pt')
//! shutil.copy(vec, 'models/tfidf_vectorizer_multitask.json')
//! ```
//!
//! ### Using HuggingFace CLI:
//!
//! ```bash
//! mkdir -p models && cd models
//! huggingface-cli download ElderRyan/psycial \
//!     mlp_weights_multitask.pt \
//!     tfidf_vectorizer_multitask.json
//! ```
//!
//! ## Requirements
//!
//! ### Runtime Dependencies
//!
//! - **PyTorch/libtorch**: Required for neural network inference
//!   ```bash
//!   conda install pytorch
//!   ```
//!
//! - **CUDA** (optional): For GPU acceleration
//!   ```bash
//!   conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
//!   ```
//!
//! ### Environment Variables
//!
//! ```bash
//! export LIBTORCH_USE_PYTORCH=1
//! export LIBTORCH_BYPASS_VERSION_CHECK=1
//! export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda/lib64
//! ```
//!
//! ## Model Information
//!
//! - **Type**: Multi-Task MLP with GPU acceleration
//! - **Input**: TF-IDF (5000) + BERT embeddings (384) = 5384 dimensions
//! - **Architecture**: 4 independent binary classifiers (E/I, S/N, T/F, J/P)
//! - **Model Size**: ~27 MB
//! - **Overall Accuracy**: 52.05%
//!   - E/I: 82.77%
//!   - S/N: 88.18%
//!   - T/F: 81.67%
//!   - J/P: 77.12%
//!
//! ## Performance Tips
//!
//! 1. **Reuse predictor instances**: Loading the model is expensive
//!    ```rust
//!    // Good ✓
//!    let predictor = Predictor::new()?;
//!    for text in texts {
//!        predictor.predict(text)?;
//!    }
//!
//!    // Bad ✗
//!    for text in texts {
//!        let predictor = Predictor::new()?;
//!        predictor.predict(text)?;
//!    }
//!    ```
//!
//! 2. **Use batch predictions**: More efficient for multiple texts
//!    ```rust
//!    # use psycial::api::Predictor;
//!    # let predictor = Predictor::new().unwrap();
//!    # let texts = vec!["text1", "text2"];
//!    // Efficient ✓
//!    predictor.predict_batch(&texts)?;
//!
//!    // Less efficient ✗
//!    texts.iter().map(|t| predictor.predict(t)).collect::<Result<Vec<_>, _>>()?;
//!    # Ok::<(), Box<dyn std::error::Error>>(())
//!    ```
//!
//! 3. **Enable GPU for large workloads**:
//!    ```bash
//!    cargo build --release --features gpu
//!    ```
//!
//! ## Error Handling
//!
//! ```rust
//! use psycial::api::Predictor;
//!
//! match Predictor::new() {
//!     Ok(predictor) => {
//!         match predictor.predict("text") {
//!             Ok(result) => println!("Type: {}", result.mbti_type),
//!             Err(e) => eprintln!("Prediction failed: {}", e),
//!         }
//!     }
//!     Err(e) => {
//!         eprintln!("Failed to load model: {}", e);
//!         eprintln!("\nSolutions:");
//!         eprintln!("1. Enable 'auto-download' feature");
//!         eprintln!("2. Download models to models/ directory manually");
//!         eprintln!("3. Specify custom model path with PredictorConfig");
//!     }
//! }
//! ```
//!
//! ## Examples
//!
//! See the `examples/` directory for complete examples:
//!
//! ```bash
//! # Simple usage
//! cargo run --example simple --features auto-download
//!
//! # Batch predictions
//! cargo run --example batch --features auto-download
//! ```
//!
//! ## Links
//!
//! - **GitHub**: <https://github.com/RyanKung/psycial>
//! - **Hugging Face**: <https://huggingface.co/ElderRyan/psycial>
//! - **Web Demo**: <https://psycial.0xbase.ai>
//!
//! ## License
//!
//! GNU General Public License v3.0 (GPLv3)

// Public API modules
pub mod api;
pub mod model_loader;

// Core modules
pub mod baseline;
pub mod bert_mlp;
pub mod bert_embedder;
pub mod bert_only;
pub mod hybrid;
pub mod neural_net;
pub mod neural_net_advanced;
pub mod neural_net_gpu;
pub mod neural_net_gpu_multitask;
pub mod psyattention;
pub mod psyattention_candle;
pub mod psyattention_full;

// Re-export commonly used items from hybrid module
pub use hybrid::tfidf::TfidfVectorizer;

// Test/experimental modules
pub mod test_psy_features;
pub mod test_orthogonality;
pub mod test_confidence_ensemble;

// Re-export commonly used types for convenience
use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

/// MBTI record structure from training data
#[derive(Debug, Deserialize, Clone)]
pub struct MbtiRecord {
    /// MBTI personality type (e.g., "INTJ", "ENFP")
    #[serde(rename = "type")]
    pub mbti_type: String,
    
    /// User's posts/text content
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
/// # Arguments
/// * `records` - The dataset to split
/// * `train_ratio` - Ratio of training data (0.0 to 1.0)
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
