//! Test confidence-based ensemble strategy
//! Use psychological features as fallback when main model has low confidence

use crate::hybrid::config::Config;
use crate::hybrid::data::MbtiRecord;
use crate::hybrid::tfidf::TfidfVectorizer;
use crate::hybrid::train::FeatureNormalizer;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use crate::psyattention::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use csv::ReaderBuilder;
use ndarray::Array2;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::error::Error;
use std::fs::File;

pub fn test_confidence_ensemble(threshold: f64) -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Testing Confidence-Based Ensemble                        ║");
    println!("║  Threshold: {:.2}                                           ║", threshold);
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Load data
    let config = Config::load("config.toml").unwrap_or_default();
    let file = File::open(&config.data.csv_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let mut records: Vec<MbtiRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    
    let mut rng = thread_rng();
    records.shuffle(&mut rng);
    let split = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split];
    let test_records = &records[split..];

    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();

    println!("Step 1: Training Model A (TF-IDF + BERT)...\n");

    // Train Model A
    let mut tfidf = TfidfVectorizer::new(5000);
    tfidf.fit(&train_texts);
    let bert_encoder = RustBertEncoder::new()?;
    
    let mut train_features_a = Vec::new();
    for text in &train_texts {
        let mut feat = tfidf.transform(text);
        feat.extend(bert_encoder.extract_features(text)?);
        train_features_a.push(feat);
    }
    
    let mut model_a = MultiTaskGpuMLP::new(5384, vec![1024, 768, 512, 256], 0.001, 0.5, 0.0);
    model_a.train_per_dimension_weighted(
        &train_features_a,
        &train_labels,
        vec![30, 30, 25, 30],
        vec![1.2, 1.0, 1.0, 1.3],
        64,
    );

    println!("\nStep 2: Training Model B (Psychological Features)...\n");

    // Train Model B
    let extractor = FullPsychologicalExtractor::new();
    let all_psy: Vec<Vec<f64>> = train_texts
        .iter()
        .map(|text| extractor.extract_all_features(text))
        .collect();

    let n_samples = all_psy.len();
    let mut feature_matrix = Array2::<f64>::zeros((n_samples, 930));
    for (i, features) in all_psy.iter().enumerate() {
        for (j, &value) in features.iter().enumerate() {
            feature_matrix[[i, j]] = value;
        }
    }

    let mut selector = PearsonFeatureSelector::new(0.85);
    selector.fit(&feature_matrix, 108);

    let selected_psy: Vec<Vec<f64>> = all_psy
        .into_iter()
        .map(|f| selector.transform(&f))
        .collect();

    let normalizer_b = FeatureNormalizer::fit(&selected_psy);
    let train_features_b: Vec<Vec<f64>> = selected_psy
        .iter()
        .map(|f| normalizer_b.transform(f))
        .collect();

    let mut model_b = MultiTaskGpuMLP::new(108, vec![256, 128, 64], 0.001, 0.3, 0.0);
    model_b.train_per_dimension_weighted(
        &train_features_b,
        &train_labels,
        vec![40, 40, 40, 40],
        vec![1.0; 4],
        64,
    );

    println!("\nStep 3: Evaluating with Confidence-Based Ensemble...\n");

    // Extract test features
    let mut test_features_a = Vec::new();
    for text in &test_texts {
        let mut feat = tfidf.transform(text);
        feat.extend(bert_encoder.extract_features(text)?);
        test_features_a.push(feat);
    }

    let all_test_psy: Vec<Vec<f64>> = test_texts
        .iter()
        .map(|text| extractor.extract_all_features(text))
        .collect();
    
    let test_features_b: Vec<Vec<f64>> = all_test_psy
        .into_iter()
        .map(|f| {
            let selected = selector.transform(&f);
            normalizer_b.transform(&selected)
        })
        .collect();

    // Get predictions with confidence
    let preds_conf_a = model_a.predict_batch_with_confidence(&test_features_a);
    let preds_b = model_b.predict_batch(&test_features_b);

    // Ensemble strategy: Use Model B when Model A's minimum confidence is below threshold
    let mut ensemble_predictions = Vec::new();
    let mut fallback_count = 0;

    for ((pred_a, conf_a), pred_b) in preds_conf_a.iter().zip(preds_b.iter()) {
        let min_conf = conf_a.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let final_pred = if min_conf < threshold {
            // Low confidence in Model A -> use Model B
            fallback_count += 1;
            pred_b.clone()
        } else {
            pred_a.clone()
        };
        
        ensemble_predictions.push(final_pred);
    }

    // Calculate accuracies
    let mut correct_a = 0;
    let mut correct_b = 0;
    let mut correct_ensemble = 0;

    for i in 0..test_labels.len() {
        if preds_conf_a[i].0 == test_labels[i] {
            correct_a += 1;
        }
        if preds_b[i] == test_labels[i] {
            correct_b += 1;
        }
        if ensemble_predictions[i] == test_labels[i] {
            correct_ensemble += 1;
        }
    }

    let total = test_labels.len() as f64;
    let acc_a = correct_a as f64 / total;
    let acc_b = correct_b as f64 / total;
    let acc_ensemble = correct_ensemble as f64 / total;

    println!("═══════════════════════════════════════════════════════════\n");
    println!("Results:\n");
    println!("  Model A (TF-IDF+BERT):   {:.2}% ({}/{})", acc_a * 100.0, correct_a, test_labels.len());
    println!("  Model B (Psychological): {:.2}% ({}/{})", acc_b * 100.0, correct_b, test_labels.len());
    println!("  Ensemble (threshold={:.2}): {:.2}% ({}/{})", threshold, acc_ensemble * 100.0, correct_ensemble, test_labels.len());
    println!();
    println!("  Fallback to Model B: {} times ({:.1}%)", fallback_count, fallback_count as f64 / total * 100.0);
    println!();

    let improvement = (acc_ensemble - acc_a) * 100.0;
    if improvement > 0.0 {
        println!("✅ Ensemble IMPROVES accuracy by {:.2}%!", improvement);
    } else if improvement > -0.5 {
        println!("⚖️  Ensemble shows marginal change ({:+.2}%)", improvement);
    } else {
        println!("❌ Ensemble DECREASES accuracy by {:.2}%", improvement.abs());
    }

    println!("\n═══════════════════════════════════════════════════════════\n");
    Ok(())
}

/// Scan different confidence thresholds to find optimal
pub fn scan_confidence_thresholds() -> Result<(), Box<dyn Error>> {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║  Scanning Confidence Thresholds                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    println!("Testing thresholds: 0.5, 0.6, 0.7, 0.8, 0.9\n");
    println!("This will take ~5 minutes...\n");

    let thresholds = vec![0.5, 0.6, 0.7, 0.8, 0.9];
    
    for &threshold in &thresholds {
        println!("═══════════════════════════════════════════════════════════");
        test_confidence_ensemble(threshold)?;
    }

    println!("\n📊 Summary:");
    println!("  Check which threshold gives best accuracy improvement\n");

    Ok(())
}

