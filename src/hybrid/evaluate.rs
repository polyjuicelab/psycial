//! Model evaluation and results reporting.

use super::config::Config;
use super::data::MbtiRecord;
use super::save::{save_model, save_model_singletask};
use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use crate::psyattention::bert_rustbert::RustBertEncoder;
use std::error::Error;
use std::time::Instant;

/// Context structure to pass evaluation parameters without too many arguments.
pub struct EvaluationContext<'a> {
    pub train_features: &'a [Vec<f64>],
    pub train_labels: &'a [String],
    pub test_records: &'a [MbtiRecord],
    pub tfidf: &'a TfidfVectorizer,
    pub bert_encoder: &'a RustBertEncoder,
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
    println!("| Paper Target (BERT + Transformer)          |  86.30%  |");
    println!("+--------------------------------------------+----------+\n");

    let improvement = (test_acc - 0.2173) / 0.2173 * 100.0;
    let vs_paper = test_acc / 0.8630 * 100.0;

    println!("Analysis:");
    println!("  Improvement over baseline: {:.1}%", improvement);
    println!("  Progress toward paper: {:.1}%", vs_paper);
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
