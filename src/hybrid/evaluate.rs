//! Model evaluation and results reporting.

use super::config::Config;
use super::data::MbtiRecord;
use super::save::{save_model, save_model_singletask};
use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use crate::psyattention::full_features::{FullPsychologicalExtractor, PearsonFeatureSelector};
use crate::psyattention::psychological_features::PsychologicalFeatureExtractor;
use std::error::Error;
use std::time::Instant;

/// Context structure to pass evaluation parameters without too many arguments.
pub struct EvaluationContext<'a> {
    pub train_features: &'a [Vec<f64>],
    pub train_labels: &'a [String],
    pub test_records: &'a [MbtiRecord],
    pub tfidf: &'a TfidfVectorizer,
    pub bert_encoder: &'a RustBertEncoder,
    pub psy_selector: &'a Option<PearsonFeatureSelector>,
    pub train_start: Instant,
    pub config: &'a Config,
}

/// Evaluate a multi-task model on training and test sets.
pub fn evaluate_multitask_model(
    mlp: &MultiTaskGpuMLP,
    ctx: &EvaluationContext,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\nTotal training time: {:.2}s\n",
        ctx.train_start.elapsed().as_secs_f64()
    );
    println!("===================================================================\n");

    println!("Evaluation\n");

    // Training Set Evaluation
    println!("Training Set:");
    let train_predictions = mlp.predict_batch(ctx.train_features);
    let mut correct = 0;
    for (pred, label) in train_predictions.iter().zip(ctx.train_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / ctx.train_labels.len() as f64;
    println!("  Accuracy: {:.2}%\n", train_acc * 100.0);

    // Test Set Evaluation
    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = ctx.test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = ctx
        .test_records
        .iter()
        .map(|r| r.mbti_type.clone())
        .collect();

    // Extract psychological features if enabled (must match training)
    let test_psy_features = if ctx.config.features.use_psychological_features {
        match ctx.config.features.psy_feature_type.as_str() {
            "simple" => {
                let extractor = PsychologicalFeatureExtractor::new();
                test_texts
                    .iter()
                    .map(|text| extractor.extract_features(text))
                    .collect()
            }
            "selected" => {
                let extractor = FullPsychologicalExtractor::new();
                let all_features: Vec<Vec<f64>> = test_texts
                    .iter()
                    .map(|text| extractor.extract_all_features(text))
                    .collect();
                
                // Apply saved selector
                if let Some(selector) = ctx.psy_selector {
                    all_features
                        .into_iter()
                        .map(|features| selector.transform(&features))
                        .collect()
                } else {
                    eprintln!("Warning: Psychological features enabled but selector not found!");
                    vec![vec![]; test_texts.len()]
                }
            }
            "full" => {
                let extractor = FullPsychologicalExtractor::new();
                test_texts
                    .iter()
                    .map(|text| extractor.extract_all_features(text))
                    .collect()
            }
            _ => vec![vec![]; test_texts.len()],
        }
    } else {
        vec![vec![]; test_texts.len()]
    };

    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Test: {}/{}", i + 1, test_texts.len());
        }
        let tfidf_feat = ctx.tfidf.transform(text);
        let bert_feat = ctx.bert_encoder.extract_features(text)?;
        let psy_feat = &test_psy_features[i];
        
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        combined.extend(psy_feat);
        test_features.push(combined);
    }

    let test_predictions = mlp.predict_batch(&test_features);
    
    // Calculate overall accuracy
    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    
    // Calculate per-dimension accuracy
    let mut dim_correct = [0usize; 4]; // [E/I, S/N, T/F, J/P]
    let total = test_labels.len();
    
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        let pred_chars: Vec<char> = pred.chars().collect();
        let label_chars: Vec<char> = label.chars().collect();
        
        if pred_chars.len() >= 4 && label_chars.len() >= 4 {
            // E/I dimension
            if pred_chars[0] == label_chars[0] {
                dim_correct[0] += 1;
            }
            // S/N dimension
            if pred_chars[1] == label_chars[1] {
                dim_correct[1] += 1;
            }
            // T/F dimension
            if pred_chars[2] == label_chars[2] {
                dim_correct[2] += 1;
            }
            // J/P dimension
            if pred_chars[3] == label_chars[3] {
                dim_correct[3] += 1;
            }
        }
    }
    
    println!("  Overall Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Per-Dimension Accuracy:");
    println!("    E/I: {:.2}% ({}/{})", 
        dim_correct[0] as f64 / total as f64 * 100.0, dim_correct[0], total);
    println!("    S/N: {:.2}% ({}/{})", 
        dim_correct[1] as f64 / total as f64 * 100.0, dim_correct[1], total);
    println!("    T/F: {:.2}% ({}/{})", 
        dim_correct[2] as f64 / total as f64 * 100.0, dim_correct[2], total);
    println!("    J/P: {:.2}% ({}/{})", 
        dim_correct[3] as f64 / total as f64 * 100.0, dim_correct[3], total);
    println!("  Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    print_results(test_acc);

    // Save model
    save_model(mlp, ctx.tfidf, &ctx.config.output)?;

    Ok(())
}

/// Evaluate a single-task model on training and test sets.
pub fn evaluate_singletask_model(
    mlp: &GpuMLP,
    ctx: &EvaluationContext,
) -> Result<(), Box<dyn Error>> {
    println!(
        "\nTotal training time: {:.2}s\n",
        ctx.train_start.elapsed().as_secs_f64()
    );
    println!("===================================================================\n");

    println!("Evaluation\n");

    // Training Set Evaluation
    println!("Training Set:");
    let train_predictions = mlp.predict_batch(ctx.train_features);
    let mut correct = 0;
    for (pred, label) in train_predictions.iter().zip(ctx.train_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let train_acc = correct as f64 / ctx.train_labels.len() as f64;
    println!("  Accuracy: {:.2}%\n", train_acc * 100.0);

    // Test Set Evaluation
    println!("Test Set:");
    let test_start = Instant::now();
    let test_texts: Vec<String> = ctx.test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = ctx
        .test_records
        .iter()
        .map(|r| r.mbti_type.clone())
        .collect();

    let mut test_features = Vec::new();
    for (i, text) in test_texts.iter().enumerate() {
        if (i + 1) % 500 == 0 {
            println!("  Test: {}/{}", i + 1, test_texts.len());
        }
        let tfidf_feat = ctx.tfidf.transform(text);
        let bert_feat = ctx.bert_encoder.extract_features(text)?;
        let mut combined = tfidf_feat;
        combined.extend(bert_feat);
        test_features.push(combined);
    }

    let test_predictions = mlp.predict_batch(&test_features);
    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        if pred == label {
            correct += 1;
        }
    }
    let test_acc = correct as f64 / test_labels.len() as f64;
    println!("  Accuracy: {:.2}%", test_acc * 100.0);
    println!("  Time: {:.2}s\n", test_start.elapsed().as_secs_f64());

    print_results(test_acc);

    // Save model
    save_model_singletask(mlp, ctx.tfidf, &ctx.config.output)?;

    Ok(())
}

/// Print formatted results comparing with baselines.
pub fn print_results(test_acc: f64) {
    println!("===================================================================\n");
    println!("Final Results\n");
    println!("+--------------------------------------------+----------+");
    println!("| Method                                     | Accuracy |");
    println!("+--------------------------------------------+----------+");
    println!("| Baseline (TF-IDF + Naive Bayes)            |  21.73%  |");
    println!("| BERT + MLP (single-task)                   |  31.99%  |");
    println!("| Hybrid (single-task, previous)             |  49.16%  |");
    println!(
        "| Hybrid (current)                           | {:>6.2}%  |",
        test_acc * 100.0
    );
    println!("+--------------------------------------------+----------+\n");

    let improvement = (test_acc - 0.2173) / 0.2173 * 100.0;

    println!("Analysis:");
    println!("  Improvement over baseline: {:.1}%", improvement);
    println!("  vs Random: {:.1}x better\n", test_acc / 0.0625);

    if test_acc > 0.55 {
        println!("🎉 OUTSTANDING: Multi-task approach showing excellent results!");
    } else if test_acc > 0.48 {
        println!("✅ EXCELLENT: Significant improvement!");
    } else if test_acc > 0.35 {
        println!("👍 GOOD: Hybrid approach working well");
    } else if test_acc > 0.30 {
        println!("📊 SUCCESS: Above BERT-only baseline");
    } else {
        println!("📝 NOTE: Results at expected level");
    }

    println!("\n===================================================================\n");
}
