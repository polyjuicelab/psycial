#![allow(dead_code)]

use ndarray::{Array2, Axis};
use std::f64::consts::E;

/// Attention-based encoder for psychological features
/// Based on Transformer architecture WITHOUT position encoding
/// (Since psychological features have no temporal order)

pub struct AttentionEncoder {
    n_layers: usize,
    n_heads: usize,
    d_model: usize,
    d_ff: usize,
    dropout: f64,
}

impl AttentionEncoder {
    pub fn new(n_layers: usize, n_heads: usize, d_model: usize, d_ff: usize) -> Self {
        AttentionEncoder {
            n_layers,
            n_heads,
            d_model,
            d_ff,
            dropout: 0.1,
        }
    }

    /// Forward pass through the encoder
    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut output = x.clone();

        for _ in 0..self.n_layers {
            // Multi-head attention (no position encoding!)
            let attn_output = self.multi_head_attention(&output);
            
            // Add & Norm
            output = self.layer_norm(&(output.clone() + attn_output));
            
            // Feed-forward network
            let ff_output = self.feed_forward(&output);
            
            // Add & Norm
            output = self.layer_norm(&(output + ff_output));
        }

        output
    }

    /// Multi-head attention mechanism
    fn multi_head_attention(&self, x: &Array2<f64>) -> Array2<f64> {
        let (_batch_size, _seq_len) = x.dim();
        let _d_k = self.d_model / self.n_heads;

        // Simplified multi-head attention
        // In full implementation, would split into multiple heads
        let attention_scores = self.scaled_dot_product_attention(x, x, x);
        
        attention_scores
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        query: &Array2<f64>,
        key: &Array2<f64>,
        value: &Array2<f64>,
    ) -> Array2<f64> {
        let d_k = key.shape()[1] as f64;
        
        // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        let scores = query.dot(&key.t()) / d_k.sqrt();
        let attention_weights = self.softmax(&scores, Axis(1));
        let output = attention_weights.dot(value);
        
        output
    }

    /// Softmax activation
    fn softmax(&self, x: &Array2<f64>, axis: Axis) -> Array2<f64> {
        let exp_x = x.mapv(|v| E.powf(v));
        let sum_exp = exp_x.sum_axis(axis).insert_axis(axis);
        exp_x / &sum_exp
    }

    /// Feed-forward network
    fn feed_forward(&self, x: &Array2<f64>) -> Array2<f64> {
        // FFN(x) = max(0, xW1 + b1)W2 + b2
        // Simplified: just apply ReLU activation
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    /// Layer normalization
    fn layer_norm(&self, x: &Array2<f64>) -> Array2<f64> {
        let mean = x.mean().unwrap();
        let var = x.mapv(|v| (v - mean).powi(2)).mean().unwrap();
        let std = (var + 1e-5).sqrt();
        
        x.mapv(|v| (v - mean) / std)
    }
}

/// Simpler weighted attention encoder (practical alternative)
pub struct WeightedAttentionEncoder {
    feature_weights: Vec<f64>,
}

impl WeightedAttentionEncoder {
    /// Create encoder with predefined feature weights from paper
    pub fn from_paper_weights() -> Self {
        // Weights from PsyAttention paper (Top 9 features)
        WeightedAttentionEncoder {
            feature_weights: vec![0.83, 0.78, 0.75, 0.75, 0.70, 0.68, 0.66, 0.63, 0.63],
        }
    }

    /// Create encoder with custom weights
    pub fn new(weights: Vec<f64>) -> Self {
        WeightedAttentionEncoder {
            feature_weights: weights,
        }
    }

    /// Encode features using weighted attention
    pub fn encode(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(
            features.len(),
            self.feature_weights.len(),
            "Feature count must match weight count"
        );

        // Apply attention weights
        features
            .iter()
            .zip(self.feature_weights.iter())
            .map(|(f, w)| f * w)
            .collect()
    }

    /// Encode features with softmax normalization
    pub fn encode_softmax(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(
            features.len(),
            self.feature_weights.len(),
            "Feature count must match weight count"
        );

        // Calculate attention scores
        let scores: Vec<f64> = features
            .iter()
            .zip(self.feature_weights.iter())
            .map(|(f, w)| f * w)
            .collect();

        // Apply softmax
        let exp_scores: Vec<f64> = scores.iter().map(|&s| E.powf(s)).collect();
        let sum_exp: f64 = exp_scores.iter().sum();

        exp_scores.iter().map(|&e| e / sum_exp).collect()
    }

    /// Get the attention weights
    pub fn weights(&self) -> &[f64] {
        &self.feature_weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_encoder() {
        let encoder = WeightedAttentionEncoder::from_paper_weights();
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        
        let encoded = encoder.encode(&features);
        
        assert_eq!(encoded.len(), 9);
        // Higher weight features should have higher encoded values
        assert!(encoded[0] < encoded[8]); // 0.1*0.83 < 0.9*0.63
    }

    #[test]
    fn test_softmax_encoding() {
        let encoder = WeightedAttentionEncoder::from_paper_weights();
        let features = vec![1.0; 9];
        
        let encoded = encoder.encode_softmax(&features);
        
        // Softmax output should sum to ~1.0
        let sum: f64 = encoded.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attention_encoder_creation() {
        let encoder = AttentionEncoder::new(8, 4, 64, 256);
        assert_eq!(encoder.n_layers, 8);
        assert_eq!(encoder.n_heads, 4);
    }
}

