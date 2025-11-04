#![allow(dead_code)]

/// Full psychological feature extractor
/// Integrates SEANCE (271) + TAACO (168) + TAALES (491) = 930 features
/// Implements Pearson correlation-based feature selection

use super::seance::SeanceExtractor;
use super::taaco::TaacoExtractor;
use super::taales::TaalesExtractor;
use ndarray::{Array1, Array2};

pub struct FullPsychologicalExtractor {
    seance: SeanceExtractor,
    taaco: TaacoExtractor,
    taales: TaalesExtractor,
}

impl FullPsychologicalExtractor {
    pub fn new() -> Self {
        FullPsychologicalExtractor {
            seance: SeanceExtractor::new(),
            taaco: TaacoExtractor::new(),
            taales: TaalesExtractor::new(),
        }
    }

    /// Extract all 930 psychological features from text
    pub fn extract_all_features(&self, text: &str) -> Vec<f64> {
        let mut features = Vec::with_capacity(930);
        
        // SEANCE: 271 features (emotion and sentiment)
        let seance_features = self.seance.extract_features(text);
        features.extend(seance_features);
        
        // TAACO: 168 features (cohesion)
        let taaco_features = self.taaco.extract_features(text);
        features.extend(taaco_features);
        
        // TAALES: 491 features (lexical sophistication)
        let taales_features = self.taales.extract_features(text);
        features.extend(taales_features);
        
        assert_eq!(features.len(), 930, "Should have exactly 930 features");
        features
    }

    /// Get feature category information
    pub fn get_feature_categories() -> Vec<(&'static str, usize, usize)> {
        vec![
            ("SEANCE (Emotion/Sentiment)", 0, 271),
            ("TAACO (Cohesion)", 271, 439),
            ("TAALES (Lexical Sophistication)", 439, 930),
        ]
    }
}

impl Default for FullPsychologicalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Pearson correlation-based feature selector
/// Reduces 930 features to ~102-114 features (85% reduction)
pub struct PearsonFeatureSelector {
    threshold: f64,
    selected_indices: Vec<usize>,
}

impl PearsonFeatureSelector {
    pub fn new(threshold: f64) -> Self {
        PearsonFeatureSelector {
            threshold,
            selected_indices: Vec::new(),
        }
    }

    /// Fit the selector on training data and identify features to keep
    pub fn fit(&mut self, feature_matrix: &Array2<f64>, target_features: usize) -> Vec<usize> {
        let n_features = feature_matrix.shape()[1];
        
        println!("Computing Pearson correlation matrix for {} features...", n_features);
        
        // Compute correlation matrix
        let correlation_matrix = self.compute_correlation_matrix(feature_matrix);
        
        println!("Finding highly correlated feature groups (threshold: {})...", self.threshold);
        
        // Find feature groups based on correlation
        let mut selected = Vec::new();
        let mut used = vec![false; n_features];
        
        // Greedy selection: pick features with highest variance first
        let mut variances: Vec<(usize, f64)> = (0..n_features)
            .map(|i| {
                let col = feature_matrix.column(i);
                let mean = col.mean().unwrap_or(0.0);
                let var = col.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(0.0);
                (i, var)
            })
            .collect();
        
        variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for (idx, _) in variances {
            if selected.len() >= target_features {
                break;
            }
            
            if used[idx] {
                continue;
            }
            
            // Mark this feature and highly correlated ones as used
            selected.push(idx);
            used[idx] = true;
            
            for j in 0..n_features {
                if !used[j] && correlation_matrix[[idx, j]].abs() > self.threshold {
                    used[j] = true;
                }
            }
        }
        
        selected.sort();
        self.selected_indices = selected.clone();
        
        println!("Selected {} features (reduced by {:.1}%)", 
                 selected.len(), 
                 (1.0 - selected.len() as f64 / n_features as f64) * 100.0);
        
        selected
    }

    /// Fit on simple Vec<Vec<f64>> data (convenience method)
    pub fn fit_simple(&mut self, features: &[Vec<f64>], target_features: usize) -> Vec<usize> {
        let n_samples = features.len();
        let n_features = if features.is_empty() { 0 } else { features[0].len() };
        
        let flat: Vec<f64> = features.iter().flat_map(|row| row.iter()).copied().collect();
        let feature_matrix = Array2::from_shape_vec((n_samples, n_features), flat)
            .expect("Failed to create feature matrix");
        
        self.fit(&feature_matrix, target_features)
    }
    
    /// Transform features using selected indices
    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        self.selected_indices
            .iter()
            .map(|&idx| features[idx])
            .collect()
    }

    /// Compute Pearson correlation matrix
    fn compute_correlation_matrix(&self, features: &Array2<f64>) -> Array2<f64> {
        let n_features = features.shape()[1];
        let mut corr_matrix = Array2::zeros((n_features, n_features));
        
        for i in 0..n_features {
            for j in i..n_features {
                let r = self.pearson_correlation(
                    &features.column(i).to_owned(),
                    &features.column(j).to_owned(),
                );
                corr_matrix[[i, j]] = r;
                corr_matrix[[j, i]] = r;
            }
            
            if (i + 1) % 100 == 0 {
                println!("  Processed {}/{} features", i + 1, n_features);
            }
        }
        
        corr_matrix
    }

    /// Calculate Pearson correlation coefficient
    fn pearson_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }
        
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);
        
        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;
        
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_full_feature_extraction() {
        let extractor = FullPsychologicalExtractor::new();
        let features = extractor.extract_all_features("I am happy because life is good.");
        
        assert_eq!(features.len(), 930);
    }

    #[test]
    fn test_feature_categories() {
        let categories = FullPsychologicalExtractor::get_feature_categories();
        assert_eq!(categories.len(), 3);
        assert_eq!(categories[0].1, 0);   // SEANCE starts at 0
        assert_eq!(categories[0].2, 271); // SEANCE ends at 271
        assert_eq!(categories[2].2, 930); // Total ends at 930
    }

    #[test]
    fn test_pearson_correlation() {
        let selector = PearsonFeatureSelector::new(0.85);
        
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation
        
        let corr = selector.pearson_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10, "Should have perfect correlation");
    }

    #[test]
    fn test_feature_selection() {
        let mut selector = PearsonFeatureSelector::new(0.9);
        
        // Create test feature matrix: 100 samples x 20 features
        let mut data = Vec::new();
        for _ in 0..100 {
            let mut row = Vec::new();
            for j in 0..20 {
                row.push(j as f64 + rand::random::<f64>());
            }
            data.push(row);
        }
        
        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter()).copied().collect();
        let feature_matrix = Array2::from_shape_vec((100, 20), flat).unwrap();
        
        let selected = selector.fit(&feature_matrix, 10);
        
        assert!(selected.len() <= 10);
        assert!(selected.len() > 0);
    }

    #[test]
    fn test_feature_transform() {
        let mut selector = PearsonFeatureSelector::new(0.85);
        selector.selected_indices = vec![0, 5, 10, 15];
        
        let features = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                           10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0];
        
        let transformed = selector.transform(&features);
        
        assert_eq!(transformed.len(), 4);
        assert_eq!(transformed[0], 0.0);
        assert_eq!(transformed[1], 5.0);
        assert_eq!(transformed[2], 10.0);
        assert_eq!(transformed[3], 15.0);
    }
}

