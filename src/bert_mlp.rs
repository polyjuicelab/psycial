mod psyattention;
mod neural_net;

use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

use psyattention::bert_rustbert::RustBertEncoder;
use neural_net::MLPClassifier;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: BERT + Neural Network");
    println!("  Engineering approach for maximum performance");
    println!("===================================================================\n");
    
    println!("Model: BERT embeddings (384-dim) + MLP classifier");
    println!("Target: Beat baseline (21.73%) with deep learning\n");
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

    // Initialize BERT
    println!("Initializing BERT encoder...");
    let encoder = RustBertEncoder::new()?;
    
    println!("===================================================================\n");
    println!("Extracting BERT features...");
    
    let train_start = Instant::now();
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let mut train_features = Vec::new();
    for (i, text) in train_texts.iter().enumerate() {
        if (i + 1) % 1000 == 0 {
            println!("  Train: {}/{}", i + 1, train_texts.len());
        }
        train_features.push(encoder.extract_features(text)?);
    }
    
    println!("\nTraining MLP classifier...");
    println!("  Architecture: 384 -> 256 -> 128 -> 16");
    println!("  Optimizer: SGD");
    println!("  Learning rate: 0.001");
    println!("  Epochs: 25 | Batch size: 32\n");
    
    let mut mlp = MLPClassifier::new(
        384,                    // Input: BERT embeddings
        vec![256, 128],        // Optimal depth (proven to work)
        16,                     // Output: 16 MBTI types
        0.001                   // Learning rate
    );
    
    mlp.train(&train_features, &train_labels, 25, 32);  // 25 epochs
    
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
        test_features.push(encoder.extract_features(text)?);
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
    println!("Results Summary\n");
    println!("+----------------------------------------+----------+");
    println!("| Method                                 | Accuracy |");
    println!("+----------------------------------------+----------+");
    println!("| Baseline (TF-IDF + NB)                 |  21.73%  |");
    println!("| BERT + k-NN                            |  18.39%  |");
    println!("| BERT + MLP (Neural Network)            | {:>6.2}%  |", test_acc * 100.0);
    println!("| Paper Target (BERT + Transformer)      |  86.30%  |");
    println!("+----------------------------------------+----------+\n");
    
    if test_acc > 0.2173 {
        println!("SUCCESS: Beat baseline with neural network!");
    } else if test_acc > 0.20 {
        println!("PROGRESS: Close to baseline, neural network shows promise");
    } else {
        println!("NOTE: Need more complex architecture or hyperparameter tuning");
    }
    
    println!("\n===================================================================\n");
    Ok(())
}

