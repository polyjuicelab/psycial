//! Model training pipeline for the hybrid MBTI classifier.

use super::config::Config;
use super::data::MbtiRecord;
use super::evaluate::{evaluate_multitask_model, evaluate_singletask_model, EvaluationContext};
use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use crate::psyattention::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use crate::psyattention::psychological_features::PsychologicalFeatureExtractor;
use csv::ReaderBuilder;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::time::Instant;

/// Normalization parameters for psychological features
#[derive(Serialize, Deserialize, Clone)]
pub struct FeatureNormalizer {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl FeatureNormalizer {
    pub fn fit(features: &[Vec<f64>]) -> Self {
        if features.is_empty() || features[0].is_empty() {
            return FeatureNormalizer {
                means: vec![],
                stds: vec![],
            };
        }

        let n_features = features[0].len();
        let n_samples = features.len();
        let mut means = vec![0.0; n_features];
        let mut stds = vec![0.0; n_features];

        // Compute means
        for sample in features {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n_samples as f64;
        }

        // Compute standard deviations
        for sample in features {
            for (i, &val) in sample.iter().enumerate() {
                stds[i] += (val - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / n_samples as f64).sqrt();
            if *std < 1e-10 {
                *std = 1.0; // Avoid division by zero
            }
        }

        FeatureNormalizer { means, stds }
    }

    pub fn transform(&self, features: &[f64]) -> Vec<f64> {
        features
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                if i < self.means.len() {
                    (val - self.means[i]) / self.stds[i]
                } else {
                    val
                }
            })
            .collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)?;
        println!("  ✓ Feature normalizer saved to {}", path);
        Ok(())
    }

    #[allow(dead_code)]
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(std::io::Error::other)
    }
}

/// Type alias for feature extraction result to reduce complexity.
type FeatureExtractionResult = Result<
    (
        Vec<Vec<f64>>,
        Vec<String>,
        TfidfVectorizer,
        RustBertEncoder,
        Option<PearsonFeatureSelector>,
        Option<FeatureNormalizer>,
    ),
    Box<dyn Error>,
>;

/// Parameters for model training to reduce function argument count.
struct TrainingParams<'a> {
    input_dim: usize,
    config: &'a Config,
    train_features: Vec<Vec<f64>>,
    train_labels: Vec<String>,
    test_records: &'a [MbtiRecord],
    tfidf: &'a TfidfVectorizer,
    bert_encoder: &'a RustBertEncoder,
    psy_selector: &'a Option<PearsonFeatureSelector>,
    psy_normalizer: &'a Option<FeatureNormalizer>,
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
    let (train_features, train_labels, tfidf, bert_encoder, psy_selector, psy_normalizer) =
        extract_features(train_records, &config)?;

    // Save normalizer if psychological features are used
    if let Some(ref normalizer) = psy_normalizer {
        normalizer.save("models/feature_normalizer.json").ok();
    }

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

    // Calculate input dimension based on enabled features
    let psy_dim = if config.features.use_psychological_features {
        match config.features.psy_feature_type.as_str() {
            "simple" => 9,
            "selected" => 108,
            "full" => 930,
            _ => 108, // default to selected
        }
    } else {
        0
    };

    let input_dim = config.features.max_tfidf_features + 384 + psy_dim;

    let feature_desc = if psy_dim > 0 {
        format!(
            "  Input: {} features ({} TF-IDF + 384 BERT + {} Psy)",
            input_dim, config.features.max_tfidf_features, psy_dim
        )
    } else {
        format!(
            "  Input: {} features ({} TF-IDF + 384 BERT)",
            input_dim, config.features.max_tfidf_features
        )
    };
    println!("{}", feature_desc);
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
        psy_selector: &psy_selector,
        psy_normalizer: &psy_normalizer,
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

    // Calculate input dimension including psychological features
    let psy_dim = if config.features.use_psychological_features {
        match config.features.psy_feature_type.as_str() {
            "simple" => 9,
            "selected" => 108,
            "full" => 930,
            _ => 108,
        }
    } else {
        0
    };
    let input_dim = config.features.max_tfidf_features + 384 + psy_dim;

    println!(
        "  Architecture: {} -> {:?} -> {}",
        input_dim, config.model.hidden_layers, output_desc
    );
    println!("  Learning rate: {}", config.model.learning_rate);
    println!("  Dropout: {}", config.model.dropout_rate);
    println!("  Epochs: {}", config.training.epochs);
    println!("  Batch size: {}\n", config.training.batch_size);
    println!("===================================================================\n");
}

/// Extract hybrid TF-IDF + BERT features from training data.
fn extract_features(train_records: &[MbtiRecord], config: &Config) -> FeatureExtractionResult {
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

    // Extract psychological features if enabled
    let (psy_features, psy_selector) = if config.features.use_psychological_features {
        let (features, selector) =
            extract_psychological_features(&train_texts, &config.features.psy_feature_type)?;
        (features, selector)
    } else {
        (vec![vec![]; train_texts.len()], None) // Empty features
    };

    // Normalize psychological features if enabled
    let (normalized_psy_features, psy_normalizer) = if config.features.use_psychological_features
        && !psy_features.is_empty()
        && !psy_features[0].is_empty()
    {
        println!("  Normalizing psychological features...");
        let normalizer = FeatureNormalizer::fit(&psy_features);
        let normalized: Vec<Vec<f64>> = psy_features
            .iter()
            .map(|f| normalizer.transform(f))
            .collect();
        (normalized, Some(normalizer))
    } else {
        (psy_features, None)
    };

    // Combine TF-IDF + BERT + Psychological features
    println!("  Combining features...");
    let mut train_features = Vec::new();
    for ((tfidf_feat, bert_feat), psy_feat) in tfidf_features
        .into_iter()
        .zip(all_bert_features.into_iter())
        .zip(normalized_psy_features.into_iter())
    {
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        combined.extend(psy_feat);
        train_features.push(combined);
    }

    // Report final feature dimensions and statistics
    if let Some(first) = train_features.first() {
        println!("  Final feature dimension: {}", first.len());

        // Diagnose feature value ranges
        if config.features.use_psychological_features {
            println!("\n  Feature Statistics (first sample):");
            let tfidf_end = config.features.max_tfidf_features;
            let bert_end = tfidf_end + 384;
            let psy_end = first.len();

            if first.len() >= psy_end {
                let tfidf_vals = &first[..tfidf_end];
                let bert_vals = &first[tfidf_end..bert_end];
                let psy_vals = &first[bert_end..psy_end];

                let tfidf_max = tfidf_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let bert_max = bert_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let psy_max = psy_vals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                let tfidf_min = tfidf_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let bert_min = bert_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let psy_min = psy_vals.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                println!("    TF-IDF range: [{:.4}, {:.4}]", tfidf_min, tfidf_max);
                println!("    BERT range:   [{:.4}, {:.4}]", bert_min, bert_max);
                println!("    Psy range:    [{:.4}, {:.4}]", psy_min, psy_max);

                // Check for NaN/Inf
                let has_nan = psy_vals.iter().any(|&x| x.is_nan() || x.is_infinite());
                if has_nan {
                    println!("    ⚠️  WARNING: Psychological features contain NaN/Inf!");
                }
            }
        }
    }

    Ok((
        train_features,
        train_labels,
        tfidf,
        bert_encoder,
        psy_selector,
        psy_normalizer,
    ))
}

/// Type alias for psychological feature extraction result
type PsychologicalFeaturesResult =
    Result<(Vec<Vec<f64>>, Option<PearsonFeatureSelector>), Box<dyn Error>>;

/// Extract psychological features based on configuration.
/// Returns (features, optional_selector)  
/// Note: Features are NOT normalized here - normalization happens in extract_features()
fn extract_psychological_features(
    texts: &[String],
    feature_type: &str,
) -> PsychologicalFeaturesResult {
    match feature_type {
        "simple" => {
            println!("  Extracting psychological features (simple: 9 features)...");
            let extractor = PsychologicalFeatureExtractor::new();
            let features: Vec<Vec<f64>> = texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if (i + 1) % 1000 == 0 {
                        println!("    Psychological: {}/{}", i + 1, texts.len());
                    }
                    extractor.extract_features(text)
                })
                .collect();
            Ok((features, None))
        }
        "selected" => {
            println!("  Extracting psychological features (full: 930 → 108 selected)...");
            let extractor = FullPsychologicalExtractor::new();

            // Extract all 930 features
            println!("    Extracting 930 features...");
            let all_features: Vec<Vec<f64>> = texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if (i + 1) % 500 == 0 {
                        println!("      Progress: {}/{}", i + 1, texts.len());
                    }
                    extractor.extract_all_features(text)
                })
                .collect();

            // Convert to ndarray for feature selection
            println!("    Performing feature selection (Pearson correlation)...");
            let n_samples = all_features.len();
            let n_features = 930;
            let mut feature_matrix = Array2::<f64>::zeros((n_samples, n_features));
            for (i, features) in all_features.iter().enumerate() {
                for (j, &value) in features.iter().enumerate() {
                    feature_matrix[[i, j]] = value;
                }
            }

            // Select top 108 features
            let mut selector = PearsonFeatureSelector::new(0.85);
            let selected_indices = selector.fit(&feature_matrix, 108);

            println!(
                "    Selected {} features (target: 108)",
                selected_indices.len()
            );

            // Apply feature selection
            let selected_features: Vec<Vec<f64>> = all_features
                .into_iter()
                .map(|features| selected_indices.iter().map(|&idx| features[idx]).collect())
                .collect();

            // Save selector indices for test-time use
            selector.save("models/feature_selector.json").ok();

            Ok((selected_features, Some(selector)))
        }
        "full" => {
            println!("  Extracting psychological features (full: 930 features)...");
            let extractor = FullPsychologicalExtractor::new();
            let features: Vec<Vec<f64>> = texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if (i + 1) % 500 == 0 {
                        println!("    Psychological: {}/{}", i + 1, texts.len());
                    }
                    extractor.extract_all_features(text)
                })
                .collect();
            Ok((features, None))
        }
        _ => {
            eprintln!("Unknown psychological feature type: {}", feature_type);
            eprintln!("Using default (selected)");
            extract_psychological_features(texts, "selected")
        }
    }
}

/// Train and evaluate a multi-task model.
fn train_multitask_model(params: TrainingParams) -> Result<(), Box<dyn Error>> {
    let mut mlp = MultiTaskGpuMLP::new(
        params.input_dim as i64,
        params.config.model.hidden_layers.clone(),
        params.config.model.learning_rate,
        params.config.model.dropout_rate,
        params.config.model.weight_decay,
    );

    mlp.train_per_dimension_weighted(
        &params.train_features,
        &params.train_labels,
        params.config.training.per_dimension_epochs.clone(),
        params.config.training.dimension_loss_weights.clone(),
        params.config.training.batch_size,
    );

    let eval_ctx = EvaluationContext {
        train_features: &params.train_features,
        train_labels: &params.train_labels,
        test_records: params.test_records,
        tfidf: params.tfidf,
        bert_encoder: params.bert_encoder,
        psy_selector: params.psy_selector,
        psy_normalizer: params.psy_normalizer,
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
        psy_selector: params.psy_selector,
        psy_normalizer: params.psy_normalizer,
        train_start: params.train_start,
        config: params.config,
    };
    evaluate_singletask_model(&mlp, &eval_ctx)?;
    Ok(())
}
