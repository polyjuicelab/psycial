#![allow(dead_code)]

/// BERT-integrated MBTI classifier (Rust API with libtorch backend)

use super::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use super::fusion_layer::DynamicFusionLayer;
use super::bert_rustbert::RustBertEncoder;
use std::collections::HashMap;
use std::error::Error;

pub struct BertClassifier {
    psy_extractor: FullPsychologicalExtractor,
    feature_selector: Option<PearsonFeatureSelector>,
    bert_encoder: Option<RustBertEncoder>,
    fusion_layer: DynamicFusionLayer,
    class_centroids: HashMap<String, Vec<f64>>,
    class_priors: HashMap<String, f64>,
    n_selected_features: usize,
}

impl BertClassifier {
    pub fn new(n_selected_features: usize) -> Self {
        BertClassifier {
            psy_extractor: FullPsychologicalExtractor::new(),
            feature_selector: None,
            bert_encoder: None,
            fusion_layer: DynamicFusionLayer::new(n_selected_features, 768),  // bert-base uses 768
            class_centroids: HashMap::new(),
            class_priors: HashMap::new(),
            n_selected_features,
        }
    }
    
    pub fn init_bert(&mut self) -> Result<(), Box<dyn Error>> {
        println!("🔧 Initializing Real BERT (rust-bert)...");
        println!("   Note: First run will download model (~90MB)");
        println!("   This may take 1-2 minutes...\n");
        
        match RustBertEncoder::new() {
            Ok(encoder) => {
                self.bert_encoder = Some(encoder);
                println!("   ✓ BERT initialized successfully!\n");
                Ok(())
            }
            Err(e) => {
                Err(format!("Failed to initialize BERT: {}\n\n\
                    Possible solutions:\n\
                    1. Ensure internet connection (for first download)\n\
                    2. Install libtorch if needed\n\
                    3. Check disk space (~200MB needed)\n\
                    4. Or run baseline: cargo run --release --bin baseline", e).into())
            }
        }
    }
    
    pub fn train(&mut self, texts: &[String], labels: &[String]) -> Result<(), Box<dyn Error>> {
        println!("\n╔════════════════════════════════════════════════════════╗");
        println!("║  BERT-Integrated PsyAttention Training (Pure Rust)     ║");
        println!("╚════════════════════════════════════════════════════════╝\n");
        
        // Extract psychological features
        println!("📊 Extracting 930 psychological features...");
        let mut all_psy: Vec<Vec<f64>> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            if (i + 1) % 1000 == 0 {
                println!("   {}/{}", i + 1, texts.len());
            }
            all_psy.push(self.psy_extractor.extract_all_features(text));
        }
        
        // Feature selection
        println!("\n🎯 Pearson feature selection (930 → {})...", self.n_selected_features);
        let mut selector = PearsonFeatureSelector::new(0.85);
        selector.fit_simple(&all_psy, self.n_selected_features);
        
        let selected_psy: Vec<Vec<f64>> = all_psy
            .iter()
            .map(|f| selector.transform(f))
            .collect();
        
        self.feature_selector = Some(selector);
        println!("   ✓ Selected {} features\n", self.n_selected_features);
        
        // Extract BERT features
        println!("🤖 Extracting BERT features (768 dim)...");
        let mut bert_features = Vec::new();
        
        if let Some(encoder) = &self.bert_encoder {
            for (i, text) in texts.iter().enumerate() {
                if (i + 1) % 100 == 0 {
                    println!("   {}/{}", i + 1, texts.len());
                }
                bert_features.push(encoder.extract_features(text)?);
            }
            println!("   ✓ BERT features extracted\n");
        } else {
            return Err("BERT encoder not initialized".into());
        }
        
        // Learn fusion
        println!("🔗 Learning fusion weights...");
        self.fusion_layer.learn_weights(&selected_psy, &bert_features, labels);
        
        // Fuse features
        println!("⚡ Fusing features...");
        let fused: Vec<Vec<f64>> = selected_psy
            .iter()
            .zip(bert_features.iter())
            .map(|(psy, bert)| self.fusion_layer.forward(psy, bert))
            .collect();
        
        // Train classifier
        println!("🎓 Training classifier...");
        self.train_on_fused(&fused, labels);
        
        println!("\n✅ Training complete!\n");
        Ok(())
    }
    
    fn train_on_fused(&mut self, fused: &[Vec<f64>], labels: &[String]) {
        // Normalize all features first (L2 norm)
        let normalized: Vec<Vec<f64>> = fused
            .iter()
            .map(|f| self.l2_normalize(f))
            .collect();
        
        // Calculate class priors
        let mut counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        
        let n = labels.len() as f64;
        for (class, count) in &counts {
            self.class_priors.insert(class.clone(), *count as f64 / n);
        }
        
        // Calculate normalized centroids
        for class in counts.keys() {
            let class_samples: Vec<&Vec<f64>> = normalized
                .iter()
                .zip(labels.iter())
                .filter(|(_, l)| *l == class)
                .map(|(f, _)| f)
                .collect();
            
            if class_samples.is_empty() {
                continue;
            }
            
            let dim = class_samples[0].len();
            let mut centroid = vec![0.0; dim];
            
            for sample in &class_samples {
                for (i, &val) in sample.iter().enumerate() {
                    centroid[i] += val;
                }
            }
            
            for c in &mut centroid {
                *c /= class_samples.len() as f64;
            }
            
            // Normalize centroid too
            let normalized_centroid = self.l2_normalize(&centroid);
            self.class_centroids.insert(class.clone(), normalized_centroid);
        }
    }
    
    fn l2_normalize(&self, features: &[f64]) -> Vec<f64> {
        let norm = features.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-10);
        features.iter().map(|&x| x / norm).collect()
    }
    
    fn cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        dot  // Already normalized, so dot product = cosine similarity
    }
    
    pub fn predict(&self, text: &str) -> Result<String, Box<dyn Error>> {
        // Extract and select psy features
        let mut psy = self.psy_extractor.extract_all_features(text);
        if let Some(selector) = &self.feature_selector {
            psy = selector.transform(&psy);
        }
        
        // Extract BERT features
        let bert = if let Some(encoder) = &self.bert_encoder {
            encoder.extract_features(text)?
        } else {
            return Err("BERT not initialized".into());
        };
        
        // Fuse and normalize
        let fused = self.fusion_layer.forward(&psy, &bert);
        let normalized_fused = self.l2_normalize(&fused);
        
        // Classify using cosine similarity (better for high-dim normalized features)
        let mut best_class = String::new();
        let mut best_similarity = f64::NEG_INFINITY;
        
        for (class, centroid) in &self.class_centroids {
            // Cosine similarity (higher is better)
            let similarity = self.cosine_similarity(&normalized_fused, centroid);
            
            // Apply prior boost (favor larger classes)
            let adjusted = if let Some(&prior) = self.class_priors.get(class) {
                similarity + prior.ln().abs() * 0.15  // Logarithmic prior boost
            } else {
                similarity
            };
            
            if adjusted > best_similarity {
                best_similarity = adjusted;
                best_class = class.clone();
            }
        }
        
        Ok(best_class)
    }
}

