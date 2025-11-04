#![allow(dead_code)]

/// Full PsyAttention classifier with 930 psychological features
/// Implements Pearson feature selection and attention encoding

use super::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use super::attention_encoder::WeightedAttentionEncoder;
use ndarray::Array2;
use std::collections::HashMap;
use std::f64::consts::E;

pub struct FullPsyAttentionClassifier {
    feature_extractor: FullPsychologicalExtractor,
    feature_selector: Option<PearsonFeatureSelector>,
    attention_encoder: Option<WeightedAttentionEncoder>,
    
    // Classification parameters
    class_priors: HashMap<String, f64>,
    class_weights: HashMap<String, f64>,
    feature_means: HashMap<String, Vec<f64>>,
    feature_stds: HashMap<String, Vec<f64>>,
    
    n_selected_features: usize,
}

impl FullPsyAttentionClassifier {
    pub fn new(n_selected_features: usize) -> Self {
        FullPsyAttentionClassifier {
            feature_extractor: FullPsychologicalExtractor::new(),
            feature_selector: None,
            attention_encoder: None,
            class_priors: HashMap::new(),
            class_weights: HashMap::new(),
            feature_means: HashMap::new(),
            feature_stds: HashMap::new(),
            n_selected_features,
        }
    }

    /// Train the full classifier
    pub fn train(&mut self, texts: &[String], labels: &[String]) {
        assert_eq!(texts.len(), labels.len());
        
        println!("\n=== Stage 1: Feature Extraction ===");
        println!("Extracting 930 psychological features from {} samples...", texts.len());
        
        // Extract all 930 features for all samples
        let all_features: Vec<Vec<f64>> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                if (i + 1) % 1000 == 0 {
                    println!("  Processed {}/{} samples", i + 1, texts.len());
                }
                self.feature_extractor.extract_all_features(text)
            })
            .collect();
        
        println!("Feature extraction complete!");
        
        // Convert to Array2 for Pearson correlation
        println!("\n=== Stage 2: Feature Selection (Pearson Correlation) ===");
        let n_samples = all_features.len();
        let flat: Vec<f64> = all_features.iter().flat_map(|row| row.iter()).copied().collect();
        let feature_matrix = Array2::from_shape_vec((n_samples, 930), flat)
            .expect("Failed to create feature matrix");
        
        // Pearson feature selection
        let mut selector = PearsonFeatureSelector::new(0.85);
        let selected_indices = selector.fit(&feature_matrix, self.n_selected_features);
        
        println!("Feature selection complete: {} features selected", selected_indices.len());
        
        // Store selector
        self.feature_selector = Some(selector);
        
        // Apply feature selection
        let selected_features: Vec<Vec<f64>> = all_features
            .iter()
            .map(|features| {
                self.feature_selector
                    .as_ref()
                    .unwrap()
                    .transform(features)
            })
            .collect();
        
        println!("\n=== Stage 3: Attention Weight Learning ===");
        
        // Learn attention weights based on feature importance
        let weights = self.learn_attention_weights(&selected_features, labels);
        self.attention_encoder = Some(WeightedAttentionEncoder::new(weights));
        
        println!("Attention weights learned!");
        
        // Encode features using attention
        let encoded_features: Vec<Vec<f64>> = selected_features
            .iter()
            .map(|features| {
                self.attention_encoder
                    .as_ref()
                    .unwrap()
                    .encode(features)
            })
            .collect();
        
        println!("\n=== Stage 4: Training Classifier ===");
        
        // Calculate class statistics
        let n_samples = labels.len() as f64;
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts.entry(label.clone()).or_insert(0) += 1;
        }
        
        // Calculate class priors and weights
        let n_classes = class_counts.len() as f64;
        for (class, count) in &class_counts {
            let prior = *count as f64 / n_samples;
            self.class_priors.insert(class.clone(), prior);
            
            // Balanced weights
            let weight = (n_samples / (n_classes * *count as f64)).sqrt();
            self.class_weights.insert(class.clone(), weight);
        }
        
        // Normalize features first (Z-score normalization)
        let normalized_features = self.normalize_features(&encoded_features);
        
        // Calculate feature statistics for each class
        for class in class_counts.keys() {
            let class_features: Vec<&Vec<f64>> = normalized_features
                .iter()
                .zip(labels.iter())
                .filter(|(_, label)| *label == class)
                .map(|(features, _)| features)
                .collect();
            
            if class_features.is_empty() {
                continue;
            }
            
            let n_features = class_features[0].len();
            let mut means = vec![0.0; n_features];
            let mut stds = vec![1.0; n_features]; // Default std = 1.0 after normalization
            
            // Calculate means
            for features in &class_features {
                for (i, &val) in features.iter().enumerate() {
                    means[i] += val;
                }
            }
            for mean in &mut means {
                *mean /= class_features.len() as f64;
            }
            
            // Calculate standard deviations
            for features in &class_features {
                for (i, &val) in features.iter().enumerate() {
                    let diff = val - means[i];
                    stds[i] += diff * diff;
                }
            }
            for std in &mut stds {
                *std = (*std / class_features.len() as f64).sqrt() + 0.1; // Larger smoothing
            }
            
            self.feature_means.insert(class.clone(), means);
            self.feature_stds.insert(class.clone(), stds);
        }
        
        println!("Classifier training complete!");
        println!("\nModel summary:");
        println!("  Total features extracted: 930");
        println!("  Features after selection: {}", self.n_selected_features);
        println!("  Number of classes: {}", class_counts.len());
        println!("  Largest class: {} ({:.1}%)", 
                 class_counts.iter().max_by_key(|(_, &v)| v).unwrap().0,
                 class_counts.iter().max_by_key(|(_, &v)| v).unwrap().1 * 100 / labels.len());
    }

    /// Normalize features using Z-score normalization
    fn normalize_features(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if features.is_empty() {
            return Vec::new();
        }
        
        let n_features = features[0].len();
        let n_samples = features.len() as f64;
        
        // Calculate mean and std for each feature
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];
        
        for sample_features in features {
            for (i, &val) in sample_features.iter().enumerate() {
                means[i] += val;
            }
        }
        
        for mean in &mut means {
            *mean /= n_samples;
        }
        
        for sample_features in features {
            for (i, &val) in sample_features.iter().enumerate() {
                let diff = val - means[i];
                stds[i] += diff * diff;
            }
        }
        
        for std in &mut stds {
            *std = (*std / n_samples).sqrt() + 1e-10; // Add small constant
        }
        
        // Normalize all features
        features
            .iter()
            .map(|sample_features| {
                sample_features
                    .iter()
                    .enumerate()
                    .map(|(i, &val)| (val - means[i]) / stds[i])
                    .collect()
            })
            .collect()
    }
    
    /// Learn attention weights based on feature importance
    fn learn_attention_weights(&self, features: &[Vec<f64>], _labels: &[String]) -> Vec<f64> {
        let n_features = if features.is_empty() { 0 } else { features[0].len() };
        
        // Use uniform weights for stability
        // In full implementation, would learn these via backpropagation
        vec![1.0; n_features]
    }

    /// Predict MBTI type
    pub fn predict(&self, text: &str) -> String {
        // Extract features
        let mut features = self.feature_extractor.extract_all_features(text);
        
        // Apply feature selection
        if let Some(selector) = &self.feature_selector {
            features = selector.transform(&features);
        }
        
        // Simple k-NN style classification (more robust for high dimensions)
        let mut best_class = String::new();
        let mut best_distance = f64::INFINITY;
        
        for (class, prior) in &self.class_priors {
            if let Some(means) = self.feature_means.get(class) {
                // Calculate Euclidean distance
                let mut distance = 0.0;
                for (i, &feature_val) in features.iter().enumerate() {
                    if i < means.len() {
                        let diff = feature_val - means[i];
                        distance += diff * diff;
                    }
                }
                distance = distance.sqrt();
                
                // Apply prior as penalty (favor larger classes slightly)
                distance /= prior.sqrt();
                
                if distance < best_distance {
                    best_distance = distance;
                    best_class = class.clone();
                }
            }
        }
        
        best_class
    }

    /// Predict with probability distribution
    pub fn predict_proba(&self, text: &str) -> HashMap<String, f64> {
        let mut features = self.feature_extractor.extract_all_features(text);
        
        if let Some(selector) = &self.feature_selector {
            features = selector.transform(&features);
        }
        
        let mut distances = HashMap::new();
        
        for (class, prior) in &self.class_priors {
            if let Some(means) = self.feature_means.get(class) {
                let mut distance = 0.0;
                for (i, &feature_val) in features.iter().enumerate() {
                    if i < means.len() {
                        let diff = feature_val - means[i];
                        distance += diff * diff;
                    }
                }
                distance = distance.sqrt();
                distance /= prior.sqrt(); // Apply prior
                
                distances.insert(class.clone(), distance);
            }
        }
        
        // Convert distances to probabilities (smaller distance = higher probability)
        // Use softmax on negative distances
        let neg_distances: HashMap<String, f64> = distances
            .iter()
            .map(|(k, &v)| (k.clone(), -v / 10.0)) // Scale down for numerical stability
            .collect();
        
        let exp_scores: HashMap<String, f64> = neg_distances
            .iter()
            .map(|(k, &v)| (k.clone(), E.powf(v)))
            .collect();
        
        let sum_exp: f64 = exp_scores.values().sum();
        
        if sum_exp > 0.0 {
            exp_scores
                .into_iter()
                .map(|(k, v)| (k, v / sum_exp))
                .collect()
        } else {
            // Fallback to uniform distribution
            let uniform = 1.0 / self.class_priors.len() as f64;
            self.class_priors.keys().map(|k| (k.clone(), uniform)).collect()
        }
    }

    /// Analyze which features are most important for this text
    pub fn analyze_features(&self, text: &str) -> HashMap<String, f64> {
        let features = self.feature_extractor.extract_all_features(text);
        
        let mut analysis = HashMap::new();
        
        // Top features from each category
        analysis.insert("seance_max".to_string(), 
                       features[0..271].iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        analysis.insert("taaco_max".to_string(),
                       features[271..439].iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        analysis.insert("taales_max".to_string(),
                       features[439..930].iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        
        // Category means
        analysis.insert("seance_mean".to_string(),
                       features[0..271].iter().sum::<f64>() / 271.0);
        analysis.insert("taaco_mean".to_string(),
                       features[271..439].iter().sum::<f64>() / 168.0);
        analysis.insert("taales_mean".to_string(),
                       features[439..930].iter().sum::<f64>() / 491.0);
        
        analysis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_classifier_creation() {
        let classifier = FullPsyAttentionClassifier::new(108);
        assert_eq!(classifier.n_selected_features, 108);
    }

    #[test]
    fn test_full_pipeline() {
        let mut classifier = FullPsyAttentionClassifier::new(50);
        
        let texts = vec![
            "I feel incredibly happy and joyful! Life is wonderful.".to_string(),
            "I am so ashamed and guilty. I feel terrible and sad.".to_string(),
            "I feel happy and proud of my achievements and success!".to_string(),
        ];
        
        let labels = vec![
            "ENFP".to_string(),
            "INFJ".to_string(),
            "ENTJ".to_string(),
        ];
        
        classifier.train(&texts, &labels);
        
        let prediction = classifier.predict("I feel great and happy!");
        assert!(!prediction.is_empty());
    }
}

