//! Model training pipeline for the hybrid MBTI classifier.

use super::config::Config;
use super::data::MbtiRecord;
use super::evaluate::{evaluate_multitask_model, evaluate_singletask_model, EvaluationContext};
use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

/// Type alias for feature extraction result to reduce complexity.
type FeatureExtractionResult =
    Result<(Vec<Vec<f64>>, Vec<String>, TfidfVectorizer, RustBertEncoder), Box<dyn Error>>;

/// Parameters for model training to reduce function argument count.
struct TrainingParams<'a> {
    input_dim: usize,
    config: &'a Config,
    train_features: Vec<Vec<f64>>,
    train_labels: Vec<String>,
    test_records: &'a [MbtiRecord],
    tfidf: &'a TfidfVectorizer,
    bert_encoder: &'a RustBertEncoder,
    train_start: Instant,
}

/// Train the hybrid MBTI classifier model.
///
/// This function orchestrates the entire training pipeline:
/// 1. Load and split data
/// 2. Extract TF-IDF and BERT features
/// 3. Train GPU-accelerated MLP (single-task or multi-task)
/// 4. Evaluate and save the model
///
/// # Arguments
///
/// * `model_type_override` - Optional model type override from command line
///   ("multitask" or "single"). If None, uses config.toml setting.
pub fn train_model(model_type_override: Option<&str>) -> Result<(), Box<dyn Error>> {
    // Load configuration
    let mut config = Config::load("config.toml").unwrap_or_else(|e| {
        eprintln!("Warning: Could not load config.toml: {}", e);
        eprintln!("Using default configuration\n");
        Config::default()
    });

    // Apply command-line override if provided
    if let Some(model_type) = model_type_override {
        config.model.model_type = model_type.to_string();
        println!(
            "ℹ️  Model type overridden by command-line flag: {}\n",
            model_type
        );
    }

    print_training_header(&config);

    // Load data
    println!("Loading dataset...");
    let start = Instant::now();
    let file = File::open(&config.data.csv_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!(
        "  Loaded {} records ({:.2}s)\n",
        records.len(),
        start.elapsed().as_secs_f64()
    );

    // Shuffle and split
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * config.data.train_split) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    println!(
        "Train: {} | Test: {}\n",
        train_records.len(),
        test_records.len()
    );
    println!("===================================================================\n");

    // Extract features
    let (train_features, train_labels, tfidf, bert_encoder) =
        extract_features(train_records, &config)?;

    // Train model
    let output_desc = if config.model.model_type == "multitask" {
        "4×2"
    } else {
        "16"
    };

    println!(
        "\nTraining GPU-Accelerated MLP ({})...",
        config.model.model_type
    );
    let input_dim = config.features.max_tfidf_features + 384;
    println!(
        "  Input: {} features ({} TF-IDF + 384 BERT)",
        input_dim, config.features.max_tfidf_features
    );
    println!(
        "  Architecture: {} -> {:?} -> {}",
        input_dim, config.model.hidden_layers, output_desc
    );
    println!("  Optimizer: Adam");
    println!("  Learning rate: {}", config.model.learning_rate);
    println!("  Dropout: {}", config.model.dropout_rate);
    println!("  Epochs: {}\n", config.training.epochs);

    let train_start = Instant::now();

    // Choose model based on configuration
    let params = TrainingParams {
        input_dim,
        config: &config,
        train_features,
        train_labels,
        test_records,
        tfidf: &tfidf,
        bert_encoder: &bert_encoder,
        train_start,
    };

    if config.model.model_type == "multitask" {
        train_multitask_model(params)?;
    } else {
        train_singletask_model(params)?;
    }

    Ok(())
}

/// Print training header with configuration details.
fn print_training_header(config: &Config) {
    println!("\n===================================================================");
    println!("  MBTI Classifier: GPU-Accelerated Hybrid Model");
    println!("  TF-IDF + BERT + GPU MLP");
    println!("===================================================================\n");

    println!("Configuration:");
    println!("  Data: {}", config.data.csv_path);
    println!(
        "  Train/Test split: {:.0}%/{:.0}%",
        config.data.train_split * 100.0,
        (1.0 - config.data.train_split) * 100.0
    );
    println!("  TF-IDF features: {}", config.features.max_tfidf_features);
    println!(
        "  Model type: {} {}",
        config.model.model_type,
        if config.model.model_type == "multitask" {
            "(4 binary classifiers: E/I, S/N, T/F, J/P)"
        } else {
            "(16-way classification)"
        }
    );
    let output_desc = if config.model.model_type == "multitask" {
        "4×2"
    } else {
        "16"
    };
    println!(
        "  Architecture: {} -> {:?} -> {}",
        config.features.max_tfidf_features + 384,
        config.model.hidden_layers,
        output_desc
    );
    println!("  Learning rate: {}", config.model.learning_rate);
    println!("  Dropout: {}", config.model.dropout_rate);
    println!("  Epochs: {}", config.training.epochs);
    println!("  Batch size: {}\n", config.training.batch_size);
    println!("===================================================================\n");
}

/// Extract hybrid TF-IDF + BERT features from training data.
fn extract_features(train_records: &[MbtiRecord], _config: &Config) -> FeatureExtractionResult {
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

    // Extract TF-IDF features first
    println!("  Extracting TF-IDF features...");
    let tfidf_features: Vec<Vec<f64>> = train_texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if (i + 1) % 1000 == 0 {
                println!("    TF-IDF: {}/{}", i + 1, train_texts.len());
            }
            tfidf.transform(text)
        })
        .collect();

    // Batch extract BERT features (GPU optimized!)
    println!("  Extracting BERT features (GPU batch mode)...");
    let batch_size = 64; // Process 64 texts at once on GPU
    let mut all_bert_features = Vec::new();

    for (batch_idx, chunk) in train_texts.chunks(batch_size).enumerate() {
        let chunk_vec: Vec<String> = chunk.to_vec();
        let bert_batch = bert_encoder.extract_features_batch(&chunk_vec)?;
        all_bert_features.extend(bert_batch);

        if (batch_idx + 1) % 10 == 0 {
            println!(
                "    BERT batch: {}/{}",
                (batch_idx + 1) * batch_size,
                train_texts.len()
            );
        }
    }

    // Combine TF-IDF + BERT features
    println!("  Combining features...");
    let mut train_features = Vec::new();
    for (tfidf_feat, bert_feat) in tfidf_features
        .into_iter()
        .zip(all_bert_features.into_iter())
    {
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        train_features.push(combined);
    }

    Ok((train_features, train_labels, tfidf, bert_encoder))
}

/// Train and evaluate a multi-task model.
fn train_multitask_model(params: TrainingParams) -> Result<(), Box<dyn Error>> {
    let mut mlp = MultiTaskGpuMLP::new(
        params.input_dim as i64,
        params.config.model.hidden_layers.clone(),
        params.config.model.learning_rate,
        params.config.model.dropout_rate,
    );

    mlp.train(
        &params.train_features,
        &params.train_labels,
        params.config.training.epochs,
        params.config.training.batch_size,
    );

    let eval_ctx = EvaluationContext {
        train_features: &params.train_features,
        train_labels: &params.train_labels,
        test_records: params.test_records,
        tfidf: params.tfidf,
        bert_encoder: params.bert_encoder,
        train_start: params.train_start,
        config: params.config,
    };
    evaluate_multitask_model(&mlp, &eval_ctx)?;
    Ok(())
}

/// Train and evaluate a single-task model.
fn train_singletask_model(params: TrainingParams) -> Result<(), Box<dyn Error>> {
    let mut mlp = GpuMLP::new(
        params.input_dim as i64,
        params.config.model.hidden_layers.clone(),
        16, // 16 MBTI types
        params.config.model.learning_rate,
        params.config.model.dropout_rate,
    );

    mlp.train(
        &params.train_features,
        &params.train_labels,
        params.config.training.epochs,
        params.config.training.batch_size,
    );

    let eval_ctx = EvaluationContext {
        train_features: &params.train_features,
        train_labels: &params.train_labels,
        test_records: params.test_records,
        tfidf: params.tfidf,
        bert_encoder: params.bert_encoder,
        train_start: params.train_start,
        config: params.config,
    };
    evaluate_singletask_model(&mlp, &eval_ctx)?;
    Ok(())
}
