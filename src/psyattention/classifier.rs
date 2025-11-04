#![allow(dead_code)]

use super::psychological_features::PsychologicalFeatureExtractor;
use super::attention_encoder::WeightedAttentionEncoder;
use std::collections::HashMap;
use std::f64::consts::E;

/// PsyAttention classifier combining psychological and text features
pub struct PsyAttentionClassifier {
    psy_extractor: PsychologicalFeatureExtractor,
    psy_encoder: WeightedAttentionEncoder,
    
    // Naive Bayes parameters
    class_priors: HashMap<String, f64>,
    class_weights: HashMap<String, f64>,
    feature_probs: HashMap<String, HashMap<usize, f64>>,
}

impl PsyAttentionClassifier {
    pub fn new() -> Self {
        PsyAttentionClassifier {
            psy_extractor: PsychologicalFeatureExtractor::new(),
            psy_encoder: WeightedAttentionEncoder::from_paper_weights(),
            class_priors: HashMap::new(),
            class_weights: HashMap::new(),
            feature_probs: HashMap::new(),
        }
    }

    /// Train the classifier on labeled data
    pub fn train(&mut self, texts: &[String], labels: &[String]) {
        assert_eq!(texts.len(), labels.len(), "Texts and labels must have same length");

        let n_samples = labels.len() as f64;
        
        // Extract psychological features for all samples
        let psy_features: Vec<Vec<f64>> = texts
            .iter()
            .map(|text| self.psy_extractor.extract_features(text))
            .collect();

        // Encode features using attention
        let encoded_features: Vec<Vec<f64>> = psy_features
            .iter()
            .map(|features| self.psy_encoder.encode(features))
            .collect();

        // Calculate class counts and priors
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts.entry(label.clone()).or_insert(0) += 1;
        }

        let n_classes = class_counts.len() as f64;
        
        // Calculate class weights (inverse of frequency for balancing)
        for (class, count) in &class_counts {
            let weight = (n_samples / (n_classes * *count as f64)).sqrt();
            self.class_weights.insert(class.clone(), weight);
            
            let prior = *count as f64 / n_samples;
            self.class_priors.insert(class.clone(), prior);
        }

        // Calculate feature probabilities for each class
        for class in class_counts.keys() {
            let mut feature_sums = vec![0.0; 9];
            let mut class_sample_count = 0;

            for (i, label) in labels.iter().enumerate() {
                if label == class {
                    for (j, &feature_val) in encoded_features[i].iter().enumerate() {
                        feature_sums[j] += feature_val;
                    }
                    class_sample_count += 1;
                }
            }

            // Store average feature values for this class
            let mut feature_map = HashMap::new();
            for (j, sum) in feature_sums.iter().enumerate() {
                let avg = if class_sample_count > 0 {
                    sum / class_sample_count as f64
                } else {
                    0.0
                };
                feature_map.insert(j, avg + 1e-10); // Laplace smoothing
            }

            self.feature_probs.insert(class.clone(), feature_map);
        }
    }

    /// Predict MBTI type for a given text
    pub fn predict(&self, text: &str) -> String {
        // Extract and encode psychological features
        let psy_features = self.psy_extractor.extract_features(text);
        let encoded = self.psy_encoder.encode(&psy_features);

        let mut best_class = String::new();
        let mut best_score = f64::NEG_INFINITY;

        // Calculate score for each class
        for (class, prior) in &self.class_priors {
            let mut score = prior.ln();

            // Add feature likelihood
            if let Some(feature_probs) = self.feature_probs.get(class) {
                for (i, &feature_val) in encoded.iter().enumerate() {
                    if let Some(&prob) = feature_probs.get(&i) {
                        // Use Gaussian-like probability
                        let diff = (feature_val - prob).abs();
                        score -= diff * diff / 2.0; // Negative log-likelihood
                    }
                }
            }

            // Apply class weight
            if let Some(&weight) = self.class_weights.get(class) {
                score += weight.ln() * 0.3; // Gentle influence
            }

            if score > best_score {
                best_score = score;
                best_class = class.clone();
            }
        }

        best_class
    }

    /// Predict with confidence scores for all classes
    pub fn predict_proba(&self, text: &str) -> HashMap<String, f64> {
        let psy_features = self.psy_extractor.extract_features(text);
        let encoded = self.psy_encoder.encode(&psy_features);

        let mut scores = HashMap::new();

        for (class, prior) in &self.class_priors {
            let mut score = prior.ln();

            if let Some(feature_probs) = self.feature_probs.get(class) {
                for (i, &feature_val) in encoded.iter().enumerate() {
                    if let Some(&prob) = feature_probs.get(&i) {
                        let diff = (feature_val - prob).abs();
                        score -= diff * diff / 2.0;
                    }
                }
            }

            if let Some(&weight) = self.class_weights.get(class) {
                score += weight.ln() * 0.3;
            }

            scores.insert(class.clone(), score);
        }

        // Convert to probabilities using softmax
        let exp_scores: HashMap<String, f64> = scores
            .iter()
            .map(|(k, &v)| (k.clone(), E.powf(v)))
            .collect();

        let sum_exp: f64 = exp_scores.values().sum();

        exp_scores
            .into_iter()
            .map(|(k, v)| (k, v / sum_exp))
            .collect()
    }

    /// Get feature importance analysis
    pub fn analyze_features(&self, text: &str) -> HashMap<String, f64> {
        self.psy_extractor.extract_features_named(text)
    }
}

impl Default for PsyAttentionClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let classifier = PsyAttentionClassifier::new();
        assert_eq!(classifier.psy_encoder.weights().len(), 9);
    }

    #[test]
    fn test_train_and_predict() {
        let mut classifier = PsyAttentionClassifier::new();
        
        let texts = vec![
            "I feel so happy and joyful today! Life is wonderful.".to_string(),
            "I am ashamed and guilty about my actions. I feel terrible.".to_string(),
            "I feel so happy and delighted! Everything is going great.".to_string(),
            "I am so sad and depressed. Nothing seems to go right.".to_string(),
        ];
        
        let labels = vec![
            "ENFP".to_string(),
            "INFJ".to_string(),
            "ENFP".to_string(),
            "INFP".to_string(),
        ];

        classifier.train(&texts, &labels);

        // Test prediction
        let prediction = classifier.predict("I am feeling happy and content today!");
        assert!(!prediction.is_empty());

        // Test probability prediction
        let proba = classifier.predict_proba("I feel joyful and proud!");
        assert!(!proba.is_empty());
    }

    #[test]
    fn test_feature_analysis() {
        let classifier = PsyAttentionClassifier::new();
        let analysis = classifier.analyze_features("I am happy but also feel guilty.");
        
        assert!(analysis.contains_key("wellbeing"));
        assert!(analysis.contains_key("guilt"));
    }
}

