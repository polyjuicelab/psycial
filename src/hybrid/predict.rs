//! Single text prediction using trained models.

use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use std::error::Error;
use std::time::Instant;

/// Predict MBTI type for a single text input.
///
/// This loads a trained model from disk and predicts the MBTI type for the
/// given text using the hybrid TF-IDF + BERT approach.
///
/// # Arguments
///
/// * `text` - Input text to classify
///
/// # Note
///
/// Currently hardcoded to use single-task model files. Future enhancement
/// could auto-detect which model type is available.
pub fn predict_single(text: &str) -> Result<(), Box<dyn Error>> {
    println!("\n===================================================================");
    println!("  MBTI Classifier: GPU Prediction");
    println!("===================================================================\n");

    println!("Loading model...");

    // Load TF-IDF vectorizer
    let tfidf = TfidfVectorizer::load("models/tfidf_vectorizer.json")?;
    println!("  ✓ TF-IDF loaded");

    // Load MLP
    let mut mlp = GpuMLP::load("models/mlp_weights.pt", 5384, vec![1024, 512, 256], 16, 0.4)?;
    mlp.load_class_mapping("models/class_mapping.json")?;
    println!("  ✓ MLP loaded\n");

    // Initialize BERT
    println!("Initializing BERT...");
    let bert_encoder = RustBertEncoder::new()?;

    println!("\nInput text:");
    let display = if text.len() > 100 {
        &format!("{}...", &text[..100])
    } else {
        text
    };
    println!("  {}\n", display);

    println!("Processing...");
    let start = Instant::now();

    // Extract features
    let tfidf_feat = tfidf.transform(text);
    let bert_feat = bert_encoder.extract_features(text)?;
    let mut combined = tfidf_feat;
    combined.extend(bert_feat);

    // Predict
    let prediction = mlp.predict(&combined);

    println!("\n===================================================================");
    println!("  MBTI Type: {}", prediction);
    println!("  Time: {:.3}s", start.elapsed().as_secs_f64());
    println!("===================================================================\n");

    Ok(())
}
