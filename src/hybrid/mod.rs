//! Hybrid MBTI Personality Classifier
//!
//! This module implements a state-of-the-art MBTI personality type classifier using
//! a hybrid approach that combines:
//!
//! - **TF-IDF** (Term Frequency-Inverse Document Frequency) for statistical text features
//! - **BERT** (Bidirectional Encoder Representations from Transformers) for semantic embeddings
//! - **GPU-Accelerated MLP** for fast training and inference
//!
//! ## Architecture
//!
//! The classifier supports two model architectures:
//!
//! ### Single-Task Model (16-way classification)
//!
//! - Predicts one of 16 MBTI types directly (INTJ, ENFP, etc.)
//! - Architecture: `[5384 input] → [1024, 512, 256] → [16 output]`
//! - Uses cross-entropy loss
//!
//! ### Multi-Task Model (4 binary classifiers)
//!
//! - Predicts each MBTI dimension independently:
//!   - **E/I**: Extraversion vs Introversion
//!   - **S/N**: Sensing vs Intuition
//!   - **T/F**: Thinking vs Feeling
//!   - **J/P**: Judging vs Perceiving
//! - Architecture: `[5384 input] → [1024, 512, 256] → 4×[2 output]`
//! - Uses averaged binary cross-entropy loss
//! - Better generalization and typically higher accuracy
//!
//! ## Features
//!
//! - **GPU Acceleration**: 10-50x faster training using CUDA (via LibTorch)
//! - **Batch Processing**: Efficient GPU utilization for BERT inference
//! - **Configuration Management**: TOML-based configuration for easy experimentation
//! - **Model Persistence**: Save and load trained models
//! - **Command-Line Interface**: Easy-to-use CLI for training and prediction
//!
//! ## Performance
//!
//! | Model | Accuracy | Improvement |
//! |-------|----------|-------------|
//! | Baseline (TF-IDF + Naive Bayes) | 21.73% | - |
//! | Single-Task Hybrid | ~49% | +126% |
//! | Multi-Task Hybrid | ~55-60% | +150-180% |
//!
//! ## Usage
//!
//! ### Training
//!
//! ```bash
//! # Multi-task model (recommended)
//! ./target/release/psycial hybrid train --multi-task
//!
//! # Single-task model
//! ./target/release/psycial hybrid train --single-task
//! ```
//!
//! ### Prediction
//!
//! ```bash
//! ./target/release/psycial hybrid predict "I love solving complex problems and discussing abstract ideas"
//! ```
//!
//! ### As a Library
//!
//! ```rust,no_run
//! use psycial::hybrid::train::train_model;
//!
//! // Train with default configuration
//! train_model(Some("multitask")).expect("Training failed");
//! ```
//!
//! ## Configuration
//!
//! Create a `config.toml` file to customize model hyperparameters:
//!
//! ```toml
//! [data]
//! csv_path = "data/mbti_1.csv"
//! train_split = 0.8
//!
//! [features]
//! max_tfidf_features = 5000
//!
//! [model]
//! model_type = "multitask"  # or "single"
//! hidden_layers = [1024, 512, 256]
//! learning_rate = 0.001
//! dropout_rate = 0.5
//!
//! [training]
//! epochs = 25
//! batch_size = 64
//!
//! [output]
//! model_dir = "models"
//! tfidf_file = "tfidf_vectorizer.json"
//! mlp_file = "mlp_weights.pt"
//! class_mapping_file = "class_mapping.json"
//! ```
//!
//! ## Module Structure
//!
//! - [`config`] - Configuration structures and loading
//! - [`data`] - Dataset record structures
//! - [`tfidf`] - TF-IDF vectorizer implementation
//! - [`train`] - Model training pipeline
//! - [`evaluate`] - Model evaluation and metrics
//! - [`save`] - Model persistence
//! - [`predict`] - Single text prediction
//! - [`cli`] - Command-line interface

pub mod cli;
pub mod config;
pub mod data;
pub mod evaluate;
pub mod predict;
pub mod save;
pub mod tfidf;
pub mod train;

// Re-export commonly used items for external use
pub use cli::main_hybrid;
