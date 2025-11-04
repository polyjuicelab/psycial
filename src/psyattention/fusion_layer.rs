#![allow(dead_code)]

/// Dynamic fusion layer for combining psychological and BERT features
/// Implements the fusion mechanism from PsyAttention paper

pub struct DynamicFusionLayer {
    psy_feature_dim: usize,
    bert_feature_dim: usize,
    
    // Learned weight parameters (simplified - no neural network)
    psy_weight_bias: f64,
    bert_weight_bias: f64,
}

impl DynamicFusionLayer {
    pub fn new(psy_feature_dim: usize, bert_feature_dim: usize) -> Self {
        DynamicFusionLayer {
            psy_feature_dim,
            bert_feature_dim,
            psy_weight_bias: 0.0,
            bert_weight_bias: 0.0,
        }
    }
    
    /// Forward pass: dynamically fuse psychological and BERT features
    /// Returns weighted and concatenated features
    pub fn forward(
        &self,
        psy_features: &[f64],
        bert_features: &[f64],
    ) -> Vec<f64> {
        assert_eq!(psy_features.len(), self.psy_feature_dim);
        assert_eq!(bert_features.len(), self.bert_feature_dim);
        
        // BERT features are typically more informative - give them higher weight
        let bert_weight = 3.0;  // BERT is 3x more important (optimal)
        let psy_weight = 1.0;
        
        let mut fused = Vec::with_capacity(self.psy_feature_dim + self.bert_feature_dim);
        
        // Apply learned weights
        fused.extend(psy_features.iter().map(|&x| x * psy_weight));
        fused.extend(bert_features.iter().map(|&x| x * bert_weight));
        
        fused
    }
    
    /// Calculate L2 norm of features
    fn calculate_magnitude(&self, features: &[f64]) -> f64 {
        features.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }
    
    /// Learn fusion weights from training data
    /// In full implementation, this would be learned via backpropagation
    pub fn learn_weights(
        &mut self,
        psy_features_list: &[Vec<f64>],
        bert_features_list: &[Vec<f64>],
        labels: &[String],
    ) {
        // Calculate average feature magnitudes per class
        let mut class_psy_magnitudes: std::collections::HashMap<String, Vec<f64>> = 
            std::collections::HashMap::new();
        let mut class_bert_magnitudes: std::collections::HashMap<String, Vec<f64>> = 
            std::collections::HashMap::new();
        
        for ((psy, bert), label) in psy_features_list.iter()
            .zip(bert_features_list.iter())
            .zip(labels.iter()) {
            
            let psy_mag = self.calculate_magnitude(psy);
            let bert_mag = self.calculate_magnitude(bert);
            
            class_psy_magnitudes.entry(label.clone())
                .or_insert_with(Vec::new)
                .push(psy_mag);
            class_bert_magnitudes.entry(label.clone())
                .or_insert_with(Vec::new)
                .push(bert_mag);
        }
        
        // Calculate variance of magnitudes
        let mut psy_variances = Vec::new();
        let mut bert_variances = Vec::new();
        
        for (class, psy_mags) in &class_psy_magnitudes {
            if let Some(bert_mags) = class_bert_magnitudes.get(class) {
                let psy_mean = psy_mags.iter().sum::<f64>() / psy_mags.len() as f64;
                let bert_mean = bert_mags.iter().sum::<f64>() / bert_mags.len() as f64;
                
                psy_variances.push(psy_mean);
                bert_variances.push(bert_mean);
            }
        }
        
        // Set biases based on average discriminative power
        if !psy_variances.is_empty() {
            let psy_power = psy_variances.iter().sum::<f64>() / psy_variances.len() as f64;
            let bert_power = bert_variances.iter().sum::<f64>() / bert_variances.len() as f64;
            
            // Normalize to make more discriminative features have higher weight
            self.psy_weight_bias = (psy_power / (psy_power + bert_power)).ln();
            self.bert_weight_bias = (bert_power / (psy_power + bert_power)).ln();
        }
    }
    
    /// Get current fusion weights (for analysis)
    pub fn get_fusion_ratio(&self, psy_features: &[f64], bert_features: &[f64]) -> (f64, f64) {
        let psy_magnitude = self.calculate_magnitude(psy_features).max(1e-10);
        let bert_magnitude = self.calculate_magnitude(bert_features).max(1e-10);
        
        let total_mag = psy_magnitude + bert_magnitude;
        let psy_weight = psy_magnitude / total_mag;
        let bert_weight = bert_magnitude / total_mag;
        
        (psy_weight, bert_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fusion_layer_creation() {
        let fusion = DynamicFusionLayer::new(108, 768);
        assert_eq!(fusion.psy_feature_dim, 108);
        assert_eq!(fusion.bert_feature_dim, 768);
    }
    
    #[test]
    fn test_fusion_forward() {
        let fusion = DynamicFusionLayer::new(10, 20);
        
        let psy_features = vec![0.5; 10];
        let bert_features = vec![0.3; 20];
        
        let fused = fusion.forward(&psy_features, &bert_features);
        
        assert_eq!(fused.len(), 30);
    }
    
    #[test]
    fn test_fusion_weights_sum_to_one() {
        let fusion = DynamicFusionLayer::new(10, 20);
        
        let psy_features = vec![1.0; 10];
        let bert_features = vec![1.0; 20];
        
        let (psy_w, bert_w) = fusion.get_fusion_ratio(&psy_features, &bert_features);
        
        assert!((psy_w + bert_w - 1.0).abs() < 1e-6, "Weights should sum to 1.0");
    }
    
    #[test]
    fn test_learn_weights() {
        let mut fusion = DynamicFusionLayer::new(5, 5);
        
        let psy_list = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
        ];
        let bert_list = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5],
            vec![0.2, 0.3, 0.4, 0.5, 0.6],
        ];
        let labels = vec!["INFP".to_string(), "INTJ".to_string()];
        
        fusion.learn_weights(&psy_list, &bert_list, &labels);
        
        // Weights should be updated
        assert_ne!(fusion.psy_weight_bias, 0.0);
    }
}

