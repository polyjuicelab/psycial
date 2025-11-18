//! Test psychological features in isolation
//! This helps diagnose whether psychological features are useful for MBTI classification

use crate::hybrid::config::Config;
use crate::hybrid::data::MbtiRecord;
use crate::hybrid::train::FeatureNormalizer;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use crate::psyattention::psychological_features::PsychologicalFeatureExtractor;
use csv::ReaderBuilder;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;

/// Test psychological features alone (without TF-IDF or BERT)
pub fn test_psychological_features_only(feature_type: &str) -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Testing Psychological Features in Isolation              ║");
    println!("║  Feature Type: {:45} ║", feature_type);
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Load data
    println!("Loading dataset...");
    let config = Config::load("config.toml").unwrap_or_default();
    let file = File::open(&config.data.csv_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;

    println!("  Loaded {} records\n", records.len());

    // Shuffle and split
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    println!(
        "Train: {} | Test: {}\n",
        train_records.len(),
        test_records.len()
    );
    println!("═══════════════════════════════════════════════════════════\n");

    // Extract features
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();

    let (train_features, selector, normalizer) = match feature_type {
        "simple" => {
            println!("Extracting SIMPLE psychological features (9 features)...");
            let extractor = PsychologicalFeatureExtractor::new();
            let features: Vec<Vec<f64>> = train_texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if (i + 1) % 1000 == 0 {
                        println!("  Progress: {}/{}", i + 1, train_texts.len());
                    }
                    extractor.extract_features(text)
                })
                .collect();

            println!("  ✓ Extracted 9 features\n");

            // Normalize
            println!("Normalizing features...");
            let normalizer = FeatureNormalizer::fit(&features);
            let normalized: Vec<Vec<f64>> =
                features.iter().map(|f| normalizer.transform(f)).collect();

            (normalized, None, Some(normalizer))
        }
        "selected" => {
            println!("Extracting SELECTED psychological features (930 → 108)...");
            let extractor = FullPsychologicalExtractor::new();

            println!("  Extracting all 930 features...");
            let all_features: Vec<Vec<f64>> = train_texts
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    if (i + 1) % 500 == 0 {
                        println!("    Progress: {}/{}", i + 1, train_texts.len());
                    }
                    extractor.extract_all_features(text)
                })
                .collect();

            // Feature selection
            println!("  Performing Pearson feature selection...");
            let n_samples = all_features.len();
            let mut feature_matrix = Array2::<f64>::zeros((n_samples, 930));
            for (i, features) in all_features.iter().enumerate() {
                for (j, &value) in features.iter().enumerate() {
                    feature_matrix[[i, j]] = value;
                }
            }

            let mut selector = PearsonFeatureSelector::new(0.85);
            let selected_indices = selector.fit(&feature_matrix, 108);

            println!("  ✓ Selected {} features\n", selected_indices.len());

            // Apply selection
            let selected: Vec<Vec<f64>> = all_features
                .into_iter()
                .map(|features| selector.transform(&features))
                .collect();

            // Normalize
            println!("Normalizing features...");
            let normalizer = FeatureNormalizer::fit(&selected);
            let normalized: Vec<Vec<f64>> =
                selected.iter().map(|f| normalizer.transform(f)).collect();

            (normalized, Some(selector), Some(normalizer))
        }
        _ => {
            return Err("Unknown feature type".into());
        }
    };

    // Print feature statistics
    if let Some(first) = train_features.first() {
        println!("Feature dimension: {}", first.len());
        let max_val = first.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = first.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_val: f64 = first.iter().sum::<f64>() / first.len() as f64;

        println!("  Range: [{:.4}, {:.4}]", min_val, max_val);
        println!("  Mean:  {:.4}", mean_val);

        let has_nan = first.iter().any(|&x| x.is_nan() || x.is_infinite());
        if has_nan {
            println!("  ⚠️  WARNING: Contains NaN/Inf!");
        } else {
            println!("  ✓ No NaN/Inf\n");
        }
    }

    // Train multi-task model with ONLY psychological features
    println!("═══════════════════════════════════════════════════════════\n");
    println!("Training Multi-Task Model (Psychological Features ONLY)...\n");

    let input_dim = train_features[0].len() as i64;
    println!("  Input: {} features (psychological only)", input_dim);
    println!("  Architecture: {} -> [256, 128, 64] -> 4×2", input_dim);
    println!("  Learning rate: 0.001");
    println!("  Dropout: 0.3 (lower for small feature set)");
    println!("  Epochs: [40, 40, 40, 40]\n");

    let mut mlp = MultiTaskGpuMLP::new(
        input_dim,
        vec![256, 128, 64], // Smaller network for fewer features
        0.001,
        0.3, // Lower dropout
        0.0,
    );

    mlp.train_per_dimension_weighted(
        &train_features,
        &train_labels,
        vec![40, 40, 40, 40],
        vec![1.0, 1.0, 1.0, 1.0],
        64,
    );

    // Evaluate on training set
    println!("\n═══════════════════════════════════════════════════════════\n");
    println!("Evaluation\n");
    println!("Training Set:");
    let train_predictions = mlp.predict_batch(&train_features);
    let mut correct = 0;
    for (pred, label) in train_predictions.iter().zip(train_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / train_labels.len() as f64;
    println!("  Overall Accuracy: {:.2}%\n", train_acc * 100.0);

    // Extract test features
    println!("Test Set:");
    let test_features = match feature_type {
        "simple" => {
            let extractor = PsychologicalFeatureExtractor::new();
            let features: Vec<Vec<f64>> = test_texts
                .iter()
                .map(|text| extractor.extract_features(text))
                .collect();

            // Normalize with training normalizer
            if let Some(ref norm) = normalizer {
                features.iter().map(|f| norm.transform(f)).collect()
            } else {
                features
            }
        }
        "selected" => {
            let extractor = FullPsychologicalExtractor::new();
            let all_features: Vec<Vec<f64>> = test_texts
                .iter()
                .map(|text| extractor.extract_all_features(text))
                .collect();

            // Apply selector and normalize
            if let Some(ref sel) = selector {
                let selected: Vec<Vec<f64>> = all_features
                    .into_iter()
                    .map(|f| sel.transform(&f))
                    .collect();

                if let Some(ref norm) = normalizer {
                    selected.iter().map(|f| norm.transform(f)).collect()
                } else {
                    selected
                }
            } else {
                vec![vec![]; test_texts.len()]
            }
        }
        _ => vec![vec![]; test_texts.len()],
    };

    // Evaluate on test set
    let test_predictions = mlp.predict_batch(&test_features);

    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;

    // Calculate per-dimension accuracy
    let mut dim_correct = [0usize; 4];
    let total = test_labels.len();

    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        let pred_chars: Vec<char> = pred.chars().collect();
        let label_chars: Vec<char> = label.chars().collect();

        if pred_chars.len() >= 4 && label_chars.len() >= 4 {
            if pred_chars[0] == label_chars[0] {
                dim_correct[0] += 1;
            }
            if pred_chars[1] == label_chars[1] {
                dim_correct[1] += 1;
            }
            if pred_chars[2] == label_chars[2] {
                dim_correct[2] += 1;
            }
            if pred_chars[3] == label_chars[3] {
                dim_correct[3] += 1;
            }
        }
    }

    println!("  Overall Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Per-Dimension Accuracy:");
    println!(
        "    E/I: {:.2}% ({}/{})",
        dim_correct[0] as f64 / total as f64 * 100.0,
        dim_correct[0],
        total
    );
    println!(
        "    S/N: {:.2}% ({}/{})",
        dim_correct[1] as f64 / total as f64 * 100.0,
        dim_correct[1],
        total
    );
    println!(
        "    T/F: {:.2}% ({}/{})",
        dim_correct[2] as f64 / total as f64 * 100.0,
        dim_correct[2],
        total
    );
    println!(
        "    J/P: {:.2}% ({}/{})",
        dim_correct[3] as f64 / total as f64 * 100.0,
        dim_correct[3],
        total
    );

    println!("\n═══════════════════════════════════════════════════════════\n");
    println!("Analysis:\n");

    let vs_random = test_acc / 0.0625;
    let vs_baseline = test_acc / 0.2173;

    println!("  vs Random (6.25%):   {:.2}x better", vs_random);
    println!("  vs Baseline (21.73%): {:.2}x\n", vs_baseline);

    if test_acc > 0.25 {
        println!("✅ Psychological features ALONE beat baseline!");
        println!("   → They contain useful MBTI signals");
    } else if test_acc > 0.15 {
        println!("⚠️  Psychological features show some signal but weak");
        println!("   → May help when combined, but not strong alone");
    } else {
        println!("❌ Psychological features perform poorly");
        println!("   → Likely adding noise when combined with TF-IDF+BERT");
        println!("   → Recommendation: DISABLE psychological features");
    }

    println!("\n═══════════════════════════════════════════════════════════\n");

    Ok(())
}

/// Compare different normalization methods
pub fn test_normalization_methods() -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Testing Different Normalization Methods                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Load sample data
    let config = Config::load("config.toml").unwrap_or_default();
    let file = File::open(&config.data.csv_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let records: Vec<MbtiRecord> = rdr.deserialize().take(100).collect::<Result<_, _>>()?;

    let texts: Vec<String> = records.iter().map(|r| r.posts.clone()).collect();

    println!(
        "Extracting psychological features from {} samples...",
        texts.len()
    );
    let extractor = PsychologicalFeatureExtractor::new();
    let features: Vec<Vec<f64>> = texts
        .iter()
        .map(|text| extractor.extract_features(text))
        .collect();

    if let Some(first) = features.first() {
        println!("\nOriginal Features (first sample):");
        let max_val = first.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = first.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_val: f64 = first.iter().sum::<f64>() / first.len() as f64;
        let std_val: f64 = (first.iter().map(|&x| (x - mean_val).powi(2)).sum::<f64>()
            / first.len() as f64)
            .sqrt();

        println!("  Range: [{:.4}, {:.4}]", min_val, max_val);
        println!("  Mean:  {:.4}", mean_val);
        println!("  Std:   {:.4}", std_val);

        // Method 1: Z-score (current)
        println!("\nMethod 1: Z-score Normalization");
        let normalizer = FeatureNormalizer::fit(&features);
        let normalized = normalizer.transform(first);
        let max_norm = normalized.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_norm = normalized.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_norm: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;

        println!("  Range: [{:.4}, {:.4}]", min_norm, max_norm);
        println!("  Mean:  {:.4}", mean_norm);

        // Method 2: Min-Max to [0, 1]
        println!("\nMethod 2: Min-Max [0, 1]");
        let min_max_normalized: Vec<f64> = first
            .iter()
            .map(|&x| {
                if max_val - min_val > 1e-10 {
                    (x - min_val) / (max_val - min_val)
                } else {
                    0.5
                }
            })
            .collect();
        let mm_max = min_max_normalized
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mm_min = min_max_normalized
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        println!("  Range: [{:.4}, {:.4}]", mm_min, mm_max);

        // Method 3: Robust scaling (clip outliers)
        println!("\nMethod 3: Robust Scaling (clip to [-3σ, +3σ])");
        let robust_normalized: Vec<f64> = first
            .iter()
            .map(|&x| {
                let z = if std_val > 1e-10 {
                    (x - mean_val) / std_val
                } else {
                    0.0
                };
                z.clamp(-3.0, 3.0) / 3.0 // Clip and scale to [-1, 1]
            })
            .collect();
        let rb_max = robust_normalized
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rb_min = robust_normalized
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));

        println!("  Range: [{:.4}, {:.4}]", rb_min, rb_max);
    }

    println!("\n═══════════════════════════════════════════════════════════\n");
    println!("Recommendation:");
    println!("  • Z-score: Good for normal distributions");
    println!("  • Min-Max: Good for bounded features");
    println!("  • Robust: Good when outliers are present (max=3795!)\n");

    Ok(())
}
