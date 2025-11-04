/// Real BERT using rust-bert (Rust API, libtorch backend)
/// Based on: https://github.com/guillaume-be/rust-bert

#[cfg(feature = "bert")]
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};

use std::error::Error;

#[cfg(feature = "bert")]
pub struct RustBertEncoder {
    model: rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel,
}

#[cfg(feature = "bert")]
impl RustBertEncoder {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        println!("🦀 Loading Real BERT (rust-bert library)...");
        println!("   Model: all-MiniLM-L6-v2 (sentence-transformers)");
        println!("   Backend: libtorch (PyTorch C++)");
        println!("   Downloading model on first run...\n");
        
        // Use MiniLM - smaller but better for this task
        let model = SentenceEmbeddingsBuilder::remote(
            SentenceEmbeddingsModelType::AllMiniLmL12V2
        )
        .create_model()
        .map_err(|e| format!("Model creation failed: {}", e))?;
        
        println!("   ✓ BERT model ready! (384 dimensions)\n");
        
        Ok(RustBertEncoder { model })
    }
    
    /// Extract BERT sentence embeddings
    pub fn extract_features(&self, text: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        // Encode single sentence
        let embeddings = self.model
            .encode(&[text])
            .map_err(|e| format!("Encoding failed: {}", e))?;
        
        // Return first (and only) embedding as Vec<f64>
        if let Some(embedding) = embeddings.first() {
            Ok(embedding.iter().map(|&x| x as f64).collect())
        } else {
            Err("No embedding generated".into())
        }
    }
    
    /// Batch extract features for multiple texts (more efficient)
    pub fn extract_features_batch(&self, texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        
        let embeddings = self.model
            .encode(&text_refs)
            .map_err(|e| format!("Batch encoding failed: {}", e))?;
        
        Ok(embeddings
            .iter()
            .map(|emb| emb.iter().map(|&x| x as f64).collect())
            .collect())
    }
    
    pub fn embedding_dim(&self) -> usize {
        384  // all-MiniLM-L12-v2 dimension
    }
}

// Fallback
#[cfg(not(feature = "bert"))]
#[allow(dead_code)]
pub struct RustBertEncoder;

#[cfg(not(feature = "bert"))]
impl RustBertEncoder {
    #[allow(dead_code)]
    pub fn new() -> Result<Self, Box<dyn Error>> {
        Err("BERT not enabled. Compile with: cargo build --features bert".into())
    }
    
    #[allow(dead_code)]
    pub fn extract_features(&self, _text: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        Err("BERT not enabled".into())
    }
    
    #[allow(dead_code)]
    pub fn extract_features_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        Err("BERT not enabled".into())
    }
    
    #[allow(dead_code)]
    pub fn embedding_dim(&self) -> usize {
        384
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "bert")]
    #[test]
    fn test_rust_bert() {
        use super::*;
        let encoder = RustBertEncoder::new();
        if let Ok(enc) = encoder {
            let result = enc.extract_features("This is a test.");
            assert!(result.is_ok());
            if let Ok(features) = result {
                assert_eq!(features.len(), 384);
            }
        }
    }
}

