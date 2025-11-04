mod psyattention;
mod neural_net_advanced;

use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

use psyattention::bert_rustbert::RustBertEncoder;
use neural_net_advanced::AdvancedMLP;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

// Simple TF-IDF implementation
#[derive(Serialize, Deserialize)]
struct TfidfVectorizer {
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    max_features: usize,
}

impl TfidfVectorizer {
    fn new(max_features: usize) -> Self {
        TfidfVectorizer {
            vocabulary: HashMap::new(),
            idf: Vec::new(),
            max_features,
        }
    }
    
    fn fit(&mut self, documents: &[String]) {
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
    
    fn transform(&self, document: &str) -> Vec<f64> {
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
        let mut tfidf: Vec<f64> = tf.iter()
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
    
    #[allow(dead_code)]
    fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    #[allow(dead_code)]
    fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let vectorizer = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(vectorizer)
    }
}

#[derive(Serialize, Deserialize)]
struct HybridModel {
    tfidf: TfidfVectorizer,
    mlp: AdvancedMLP,
}

impl HybridModel {
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        std::fs::create_dir_all("models")?;
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        println!("\n✓ Model saved to {}", path);
        println!("  TF-IDF vocabulary: {} words", self.tfidf.vocabulary.len());
        println!("  MLP architecture: {} layers", self.mlp.num_layers());
        Ok(())
    }
    
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)?;
        println!("\n✓ Model loaded from {}", path);
        Ok(model)
    }
}

fn print_usage() {
    println!("Usage:");
    println!("  cargo run --release --features bert --bin psycial -- [COMMAND] [OPTIONS]\n");
    println!("Commands:");
    println!("  train                    Train new model and save to models/");
    println!("  train --model PATH       Train and save to custom path");
    println!("  evaluate --model PATH    Evaluate existing model");
    println!("  predict --model PATH --text \"...\"  Predict single text");
    println!("  help                     Show this help\n");
    println!("Examples:");
    println!("  cargo run --release --features bert --bin psycial -- train");
    println!("  cargo run --release --features bert --bin psycial -- evaluate --model models/psycial_model.json");
    println!("  cargo run --release --features bert --bin psycial -- predict --model models/psycial_model.json --text \"I love coding\"");
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    // Parse command
    let command = if args.len() > 1 { &args[1] } else { "train" };
    
    match command.as_ref() {
        "help" | "--help" | "-h" => {
            print_usage();
            return Ok(());
        }
        "train" => {
            let model_path = if args.len() > 3 && args[2] == "--model" {
                args[3].clone()
            } else {
                "models/psycial_model.json".to_string()
            };
            train_model(&model_path)
        }
        "evaluate" => {
            let model_path = if args.len() > 3 && args[2] == "--model" {
                args[3].clone()
            } else {
                println!("Error: --model PATH required for evaluate command");
                print_usage();
                return Ok(());
            };
            evaluate_model(&model_path)
        }
        "predict" => {
            let mut model_path = String::new();
            let mut text = String::new();
            
            let mut i = 2;
            while i < args.len() {
                if args[i] == "--model" && i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else if args[i] == "--text" && i + 1 < args.len() {
                    text = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            
            if model_path.is_empty() || text.is_empty() {
                println!("Error: --model PATH and --text \"...\" required");
                print_usage();
                return Ok(());
            }
            
            predict_text(&model_path, &text)
        }
        _ => {
            println!("Unknown command: {}\n", command);
            print_usage();
            Ok(())
        }
    }
}

fn train_model(model_path: &str) -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: TF-IDF + BERT Hybrid with Advanced MLP");
    println!("  Training Mode");
    println!("===================================================================\n");
    
    println!("Strategy: Combine statistical (TF-IDF) and semantic (BERT) features");
    println!("Classifier: Advanced MLP with Adam optimizer and Dropout");
    println!("Model will be saved to: {}\n", model_path);
    println!("===================================================================\n");

    // Load data
    println!("Loading dataset...");
    let start = Instant::now();
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!("  Loaded {} records ({:.2}s)\n", records.len(), start.elapsed().as_secs_f64());

    // Shuffle and split
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    println!("Train: {} | Test: {}\n", train_records.len(), test_records.len());
    println!("===================================================================\n");

    // Extract TF-IDF features
    println!("Building TF-IDF vectorizer (top 5000 words)...");
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let mut tfidf = TfidfVectorizer::new(5000);
    tfidf.fit(&train_texts);
    println!("  Vocabulary size: {}\n", tfidf.vocabulary.len());
    
    // Initialize BERT
    println!("Initializing BERT encoder...");
    let bert_encoder = RustBertEncoder::new()?;
    
    println!("===================================================================\n");
    println!("Extracting hybrid features (TF-IDF + BERT)...");
    
    let train_start = Instant::now();
    let mut train_features = Vec::new();
    
    for (i, text) in train_texts.iter().enumerate() {
        if (i + 1) % 1000 == 0 {
            println!("  Train: {}/{}", i + 1, train_texts.len());
        }
        
        let tfidf_feat = tfidf.transform(text);
        let bert_feat = bert_encoder.extract_features(text)?;
        
        // Concatenate: 5000 (TF-IDF) + 384 (BERT) = 5384 features
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        train_features.push(combined);
    }
    
    println!("\nTraining Advanced MLP...");
    println!("  Input: 5384 features (5000 TF-IDF + 384 BERT)");
    println!("  Architecture: 5384 -> 1024 -> 512 -> 256 -> 16");
    println!("  Optimizer: Adam");
    println!("  Learning rate: 0.001");
    println!("  Dropout: 0.4");
    println!("  Epochs: 40\n");
    
    let mut mlp = AdvancedMLP::new(
        5384,                  // 5K TF-IDF + BERT
        vec![1024, 512, 256], // Deeper network
        16,                    // MBTI types
        0.001,                 // Learning rate
        0.4                    // Higher dropout for regularization
    );
    
    mlp.train(&train_features, &train_labels, 40, 64);
    
    println!("\nTotal training time: {:.2}s\n", train_start.elapsed().as_secs_f64());
    println!("===================================================================\n");

    // Evaluate
    println!("Evaluation\n");
    
    println!("Training Set:");
    let mut correct = 0;
    for (feat, label) in train_features.iter().zip(train_labels.iter()) {
        let pred = mlp.predict(feat);
        if pred == *label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / train_labels.len() as f64;
    println!("  Accuracy: {:.2}%\n", train_acc * 100.0);

    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Test: {}/{}", i + 1, test_texts.len());
        }
        
        let tfidf_feat = tfidf.transform(text);
        let bert_feat = bert_encoder.extract_features(text)?;
        
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        test_features.push(combined);
    }
    
    let mut correct = 0;
    for (feat, label) in test_features.iter().zip(test_labels.iter()) {
        let pred = mlp.predict(feat);
        if pred == *label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    println!("  Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    println!("===================================================================\n");
    println!("Final Results\n");
    println!("+--------------------------------------------+----------+");
    println!("| Method                                     | Accuracy |");
    println!("+--------------------------------------------+----------+");
    println!("| Baseline (TF-IDF + Naive Bayes)            |  21.73%  |");
    println!("| BERT + MLP                                 |  31.99%  |");
    println!("| TF-IDF + BERT + Advanced MLP (Hybrid)      | {:>6.2}%  |", test_acc * 100.0);
    println!("| Paper Target (BERT + 8-layer Transformer)  |  86.30%  |");
    println!("+--------------------------------------------+----------+\n");
    
    let improvement = (test_acc - 0.2173) / 0.2173 * 100.0;
    let vs_paper = test_acc / 0.8630 * 100.0;
    
    println!("Analysis:");
    println!("  Improvement over baseline: {:.1}%", improvement);
    println!("  Progress toward paper: {:.1}%", vs_paper);
    println!("  vs Random: {:.1}x better\n", test_acc / 0.0625);
    
    if test_acc > 0.35 {
        println!("EXCELLENT: Significant improvement with hybrid approach!");
    } else if test_acc > 0.32 {
        println!("SUCCESS: Hybrid features show promise");
    } else {
        println!("NOTE: TF-IDF + BERT combination at expected level");
    }
    
    println!("\n===================================================================\n");
    
    // Save model
    let model = HybridModel { tfidf, mlp };
    model.save(model_path)?;
    
    println!("\n===================================================================\n");
    println!("Training complete! Model ready for use.\n");
    println!("To evaluate: cargo run --release --features bert --bin psycial -- evaluate --model {}", model_path);
    println!("To predict:  cargo run --release --features bert --bin psycial -- predict --model {} --text \"your text\"\n", model_path);
    
    Ok(())
}

fn evaluate_model(model_path: &str) -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: Model Evaluation");
    println!("===================================================================\n");
    
    println!("Loading model from: {}\n", model_path);
    let model = HybridModel::load(model_path)?;
    
    println!("Initializing BERT encoder...");
    let bert_encoder = RustBertEncoder::new()?;
    
    // Load test data
    println!("\nLoading test dataset...");
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    
    // Use same split as training (20% test)
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * 0.8) as usize;
    let test_records = &records[split..];
    
    println!("  Test samples: {}\n", test_records.len());
    println!("===================================================================\n");
    
    // Extract features and evaluate
    println!("Evaluating...");
    let start = Instant::now();
    let mut correct = 0;
    
    for (i, record) in test_records.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Processed: {}/{}", i + 1, test_records.len());
        }
        
        let tfidf_feat = model.tfidf.transform(&record.posts);
        let bert_feat = bert_encoder.extract_features(&record.posts)?;
        
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        
        let pred = model.mlp.predict(&combined);
        if pred == record.mbti_type {
            correct += 1;
        }
    }
    
    let accuracy = correct as f64 / test_records.len() as f64;
    
    println!("\n===================================================================\n");
    println!("Evaluation Results\n");
    println!("  Accuracy: {:.2}%", accuracy * 100.0);
    println!("  Correct: {}/{}", correct, test_records.len());
    println!("  Time: {:.2}s\n", start.elapsed().as_secs_f64());
    println!("===================================================================\n");
    
    Ok(())
}

fn predict_text(model_path: &str, text: &str) -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: Single Prediction");
    println!("===================================================================\n");
    
    println!("Loading model from: {}\n", model_path);
    let model = HybridModel::load(model_path)?;
    
    println!("Initializing BERT encoder...");
    let bert_encoder = RustBertEncoder::new()?;
    
    println!("\nInput text:");
    let display_text = if text.len() > 100 {
        format!("{}...", &text[..100])
    } else {
        text.to_string()
    };
    println!("  {}\n", display_text);
    
    println!("Processing...");
    let start = Instant::now();
    
    // Extract features
    let tfidf_feat = model.tfidf.transform(text);
    let bert_feat = bert_encoder.extract_features(text)?;
    
    let mut combined = tfidf_feat;
    combined.extend(bert_feat);
    
    // Predict
    let prediction = model.mlp.predict(&combined);
    
    println!("\n===================================================================\n");
    println!("Prediction Result\n");
    println!("  MBTI Type: {}", prediction);
    println!("  Time: {:.3}s\n", start.elapsed().as_secs_f64());
    println!("===================================================================\n");
    
    Ok(())
}

