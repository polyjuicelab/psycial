mod psyattention;

use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

use psyattention::bert_rustbert::RustBertEncoder;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                                                           ║");
    println!("║        🤖 MBTI Classifier - Pure BERT Features           ║");
    println!("║           Testing BERT-only performance                   ║");
    println!("║                                                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    
    println!("🎯 Hypothesis: BERT alone may perform better than combined features");
    println!("📊 Features: 384-dim BERT embeddings only\n");
    println!("═══════════════════════════════════════════════════════════\n");

    // Load data
    println!("📚 Loading dataset...");
    let start = Instant::now();
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!("   ✓ {} records ({:.2}s)\n", records.len(), start.elapsed().as_secs_f64());

    // Shuffle and split
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    println!("📊 Split: {} train / {} test\n", train_records.len(), test_records.len());
    println!("═══════════════════════════════════════════════════════════\n");

    // Initialize BERT
    println!("🤖 Initializing BERT...");
    let encoder = RustBertEncoder::new()?;
    
    println!("═══════════════════════════════════════════════════════════\n");
    println!("📊 Extracting BERT features for training set...");
    
    let train_start = Instant::now();
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let mut train_features = Vec::new();
    for (i, text) in train_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("   {}/{}", i + 1, train_texts.len());
        }
        train_features.push(encoder.extract_features(text)?);
    }
    
    println!("\n🎓 Training (cosine similarity classifier)...");
    
    // Normalize features
    let normalized_train: Vec<Vec<f64>> = train_features
        .iter()
        .map(|f| l2_normalize(f))
        .collect();
    
    // Calculate class centroids
    let mut class_centroids: HashMap<String, Vec<f64>> = HashMap::new();
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    
    for label in &train_labels {
        *class_counts.entry(label.clone()).or_insert(0) += 1;
    }
    
    for class in class_counts.keys() {
        let class_samples: Vec<&Vec<f64>> = normalized_train
            .iter()
            .zip(train_labels.iter())
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
        
        class_centroids.insert(class.clone(), l2_normalize(&centroid));
    }
    
    println!("   ✓ Training complete ({:.2}s)\n", train_start.elapsed().as_secs_f64());
    println!("═══════════════════════════════════════════════════════════\n");

    // Evaluate
    println!("📊 Evaluation\n");
    
    println!("Training Set:");
    let mut correct = 0;
    for (feat, label) in normalized_train.iter().zip(train_labels.iter()) {
        let pred = predict(&class_centroids, feat);
        if pred == *label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / train_labels.len() as f64;
    println!("   Accuracy: {:.2}%\n", train_acc * 100.0);

    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("   Extracting {}/{}", i + 1, test_texts.len());
        }
        test_features.push(encoder.extract_features(text)?);
    }
    
    let normalized_test: Vec<Vec<f64>> = test_features
        .iter()
        .map(|f| l2_normalize(f))
        .collect();
    
    let mut correct = 0;
    for (feat, label) in normalized_test.iter().zip(test_labels.iter()) {
        let pred = predict(&class_centroids, feat);
        if pred == *label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    println!("   Accuracy: {:.2}%", test_acc * 100.0);
    println!("   Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    println!("═══════════════════════════════════════════════════════════\n");
    println!("📊 Results\n");
    println!("┌────────────────────────────────────────┬──────────┐");
    println!("│ Method                                 │ Accuracy │");
    println!("├────────────────────────────────────────┼──────────┤");
    println!("│ Baseline (TF-IDF)                      │  21.73%  │");
    println!("│ BERT Only (384 dim)                    │ {:>6.2}%  │", test_acc * 100.0);
    println!("│ PsyAttention + BERT                    │  19.60%  │");
    println!("└────────────────────────────────────────┴──────────┘\n");
    
    if test_acc > 0.1960 {
        println!("✅ BERT-only performs better!");
        println!("   → BERT特征质量高于心理特征");
    } else {
        println!("📊 组合特征更好");
        println!("   → 需要改进融合策略");
    }
    
    println!();
    Ok(())
}

fn l2_normalize(features: &[f64]) -> Vec<f64> {
    let norm = features.iter().map(|&x| x * x).sum::<f64>().sqrt().max(1e-10);
    features.iter().map(|&x| x / norm).collect()
}

fn predict(centroids: &HashMap<String, Vec<f64>>, features: &[f64]) -> String {
    let mut best_class = String::new();
    let mut best_sim = f64::NEG_INFINITY;
    
    for (class, centroid) in centroids {
        let sim: f64 = features.iter().zip(centroid.iter()).map(|(&a, &b)| a * b).sum();
        if sim > best_sim {
            best_sim = sim;
            best_class = class.clone();
        }
    }
    
    best_class
}

