//! BERT Embedder module for extracting semantic embeddings from text
//!
//! This module provides a wrapper around RustBertEncoder for use in the API.

use crate::psyattention::bert_rustbert::RustBertEncoder;
use std::error::Error;

/// BERT embedder for extracting semantic features from text
pub struct BertEmbedder {
    encoder: RustBertEncoder,
}

impl BertEmbedder {
    /// Create a new BERT embedder
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let encoder = RustBertEncoder::new()?;
        Ok(BertEmbedder { encoder })
    }

    /// Extract BERT embeddings from a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        self.encoder.extract_features(text)
    }

    /// Extract BERT embeddings from multiple texts (batch processing)
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        self.encoder.extract_features_batch(texts)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.encoder.embedding_dim()
    }
}

