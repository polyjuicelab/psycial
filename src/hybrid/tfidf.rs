//! TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer implementation.
//!
//! This module provides a simple but effective TF-IDF implementation for converting
//! text documents into numerical feature vectors. It's used as part of the hybrid
//! feature extraction alongside BERT embeddings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TF-IDF vectorizer for converting text to numerical features.
///
/// This implementation:
/// - Filters words by minimum length (> 2 characters)
/// - Selects top N features by word frequency
/// - Applies TF-IDF weighting and L2 normalization
#[derive(Serialize, Deserialize)]
pub struct TfidfVectorizer {
    /// Word to feature index mapping
    pub vocabulary: HashMap<String, usize>,
    /// Inverse document frequency for each feature
    pub idf: Vec<f64>,
    /// Maximum number of features to keep
    pub max_features: usize,
}

impl TfidfVectorizer {
    /// Create a new TF-IDF vectorizer.
    ///
    /// # Arguments
    ///
    /// * `max_features` - Maximum number of features (words) to keep
    pub fn new(max_features: usize) -> Self {
        TfidfVectorizer {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            max_features,
        }
    }

    /// Fit the vectorizer on a collection of documents.
    ///
    /// This builds the vocabulary and calculates IDF values.
    ///
    /// # Arguments
    ///
    /// * `documents` - Training documents to fit on
    pub fn fit(&mut self, documents: &[String]) {
        let mut word_doc_count: HashMap<String, usize> = HashMap::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let words: Vec<String> = doc
                .to_lowercase()
                .split_whitespace()
                .filter(|w| w.len() > 2)
                .map(|s| s.to_string())
                .collect();

            let unique_words: std::collections::HashSet<_> = words.iter().collect();
            for word in unique_words {
                *word_doc_count.entry(word.clone()).or_insert(0) += 1;
                *word_freq.entry(word.clone()).or_insert(0) += 1;
            }
        }

        // Select top max_features by frequency
        let mut word_freq_vec: Vec<_> = word_freq.iter().collect();
        word_freq_vec.sort_by(|a, b| b.1.cmp(a.1));

        for (idx, (word, _)) in word_freq_vec.iter().take(self.max_features).enumerate() {
            self.vocabulary.insert((*word).clone(), idx);
        }

        // Calculate IDF
        self.idf = vec![0.0; self.vocabulary.len()];
        let n_docs = documents.len() as f64;

        for (word, &idx) in &self.vocabulary {
            let doc_freq = *word_doc_count.get(word).unwrap_or(&1) as f64;
            self.idf[idx] = (n_docs / doc_freq).ln();
        }
    }

    /// Transform a document into a TF-IDF feature vector.
    ///
    /// # Arguments
    ///
    /// * `document` - Text document to transform
    ///
    /// # Returns
    ///
    /// A normalized TF-IDF feature vector
    pub fn transform(&self, document: &str) -> Vec<f64> {
        let mut tf = vec![0.0; self.vocabulary.len()];

        let words: Vec<String> = document
            .to_lowercase()
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|s| s.to_string())
            .collect();

        for word in words {
            if let Some(&idx) = self.vocabulary.get(&word) {
                tf[idx] += 1.0;
            }
        }

        // Normalize TF
        let total: f64 = tf.iter().sum();
        if total > 0.0 {
            for val in &mut tf {
                *val /= total;
            }
        }

        // Apply IDF and L2 normalize
        let mut tfidf: Vec<f64> = tf
            .iter()
            .zip(self.idf.iter())
            .map(|(&t, &i)| t * i)
            .collect();

        let norm = tfidf.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for val in &mut tfidf {
                *val /= norm;
            }
        }

        tfidf
    }

    /// Save the vectorizer to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the vectorizer
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a vectorizer from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to load the vectorizer from
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vectorizer = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(vectorizer)
    }
}
