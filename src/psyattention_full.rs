mod psyattention;

use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Deserialize;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::time::Instant;

use psyattention::full_classifier::FullPsyAttentionClassifier;

#[derive(Debug, Deserialize, Clone)]
struct MbtiRecord {
    #[serde(rename = "type")]
    mbti_type: String,
    posts: String,
}

fn calculate_accuracy(predictions: &[String], actual: &[String]) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, act)| pred == act)
        .count();
    correct as f64 / predictions.len() as f64
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  PsyAttention - FULL Implementation (930 Features)            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    
    println!("Paper: 'PsyAttention: Psychological Attention Model'");
    println!("Authors: Baohua Zhang et al., Beijing Institute of Technology");
    println!("Published: arXiv:2312.00293v1, December 2023");
    println!("Target Accuracy: 86.30% (SOTA on MBTI Kaggle dataset)\n");
    
    println!("═══════════════════════════════════════════════════════════════\n");

    // Load data
    println!("📚 Loading MBTI Kaggle dataset...");
    let start_load = Instant::now();
    
    let file = File::open("data/mbti_1.csv")?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut records: Vec<MbtiRecord> = Vec::new();
    for result in rdr.deserialize() {
        let record: MbtiRecord = result?;
        records.push(record);
    }

    println!("   ✓ Loaded {} records in {:.2}s\n", records.len(), start_load.elapsed().as_secs_f64());

    // Shuffle data
    let mut rng = thread_rng();
    records.shuffle(&mut rng);

    // Split train/test (80/20)
    let split_idx = (records.len() as f64 * 0.8) as usize;
    let train_records = &records[..split_idx];
    let test_records = &records[split_idx..];

    println!("📊 Dataset Split:");
    println!("   Training set: {} samples (80%)", train_records.len());
    println!("   Test set:     {} samples (20%)", test_records.len());
    
    // Show class distribution
    let mut class_counts: HashMap<String, usize> = HashMap::new();
    for record in train_records {
        *class_counts.entry(record.mbti_type.clone()).or_insert(0) += 1;
    }
    let mut sorted_classes: Vec<_> = class_counts.iter().collect();
    sorted_classes.sort_by(|a, b| b.1.cmp(a.1));
    
    println!("\n📈 Class Distribution (Top 5):");
    for (class, count) in sorted_classes.iter().take(5) {
        let pct = (**count as f64 / train_records.len() as f64) * 100.0;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("   {}: {:>4} ({:>5.1}%) {}", class, count, pct, bar);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════\n");

    // Create classifier with feature selection
    println!("🔧 Configuration:");
    println!("   Full features: 930 (SEANCE=271, TAACO=168, TAALES=491)");
    println!("   Target features after Pearson selection: 108");
    println!("   Correlation threshold: 0.85");
    println!("   Attention-based encoding: Enabled\n");

    let start_train = Instant::now();
    
    let mut classifier = FullPsyAttentionClassifier::new(108);
    
    let train_texts: Vec<String> = train_records.iter().map(|r| r.posts.clone()).collect();
    let train_labels: Vec<String> = train_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    println!("🎓 Training Full PsyAttention Classifier...\n");
    classifier.train(&train_texts, &train_labels);
    
    let train_time = start_train.elapsed().as_secs_f64();
    println!("\n   ✓ Training completed in {:.2}s", train_time);

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Evaluate on training set
    println!("📊 Evaluating on Training Set...");
    let eval_start = Instant::now();
    
    let train_predictions: Vec<String> = train_texts
        .iter()
        .map(|text| classifier.predict(text))
        .collect();
    let train_accuracy = calculate_accuracy(&train_predictions, &train_labels);
    
    println!("   Training Accuracy: {:.2}%", train_accuracy * 100.0);
    println!("   Evaluation time: {:.2}s", eval_start.elapsed().as_secs_f64());

    // Evaluate on test set
    println!("\n📊 Evaluating on Test Set...");
    let test_start = Instant::now();
    
    let test_texts: Vec<String> = test_records.iter().map(|r| r.posts.clone()).collect();
    let test_labels: Vec<String> = test_records.iter().map(|r| r.mbti_type.clone()).collect();
    
    let test_predictions: Vec<String> = test_texts
        .iter()
        .map(|text| classifier.predict(text))
        .collect();
    let test_accuracy = calculate_accuracy(&test_predictions, &test_labels);
    
    println!("   Test Accuracy: {:.2}%", test_accuracy * 100.0);
    println!("   Evaluation time: {:.2}s", test_start.elapsed().as_secs_f64());

    println!("\n═══════════════════════════════════════════════════════════════\n");

    // Detailed sample analysis
    println!("🔍 Sample Predictions with Feature Analysis:\n");
    
    for i in 0..3.min(test_records.len()) {
        println!("─── Sample {} ───────────────────────────────────────────────", i + 1);
        println!("Text preview: {}...", 
                 &test_records[i].posts.chars().take(80).collect::<String>());
        println!();
        println!("Actual:    {}", test_records[i].mbti_type);
        println!("Predicted: {}", test_predictions[i]);
        println!("Match:     {}", if test_predictions[i] == test_records[i].mbti_type {
            "✓ CORRECT"
        } else {
            "✗ INCORRECT"
        });
        
        // Feature analysis
        let analysis = classifier.analyze_features(&test_records[i].posts);
        println!("\nFeature Category Analysis:");
        println!("  SEANCE  (Emotion):      {:.4}", analysis.get("seance_mean").unwrap_or(&0.0));
        println!("  TAACO   (Cohesion):     {:.4}", analysis.get("taaco_mean").unwrap_or(&0.0));
        println!("  TAALES  (Sophistication): {:.4}", analysis.get("taales_mean").unwrap_or(&0.0));
        
        // Probability distribution
        let proba = classifier.predict_proba(&test_records[i].posts);
        let mut proba_vec: Vec<_> = proba.iter().collect();
        proba_vec.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        println!("\nTop 3 Predictions:");
        for (rank, (class, prob)) in proba_vec.iter().take(3).enumerate() {
            println!("  {}. {}: {:.2}%", rank + 1, class, **prob * 100.0);
        }
        println!();
    }

    println!("═══════════════════════════════════════════════════════════════\n");

    // Final comparison
    println!("📈 Performance Summary:\n");
    println!("┌────────────────────────────────────────────┬──────────┐");
    println!("│ Implementation                             │ Accuracy │");
    println!("├────────────────────────────────────────────┼──────────┤");
    println!("│ Random Guessing (baseline)                 │   6.25%  │");
    println!("│ TF-IDF + Naive Bayes (baseline)            │  21.73%  │");
    println!("│ PsyAttention Simple (9 features)           │  21.21%  │");
    println!("│ PsyAttention Full (930→108 features)       │ {:>6.2}%  │", test_accuracy * 100.0);
    println!("│ Paper Target (full implementation + BERT) │  86.30%  │");
    println!("└────────────────────────────────────────────┴──────────┘");
    
    println!("\n📝 Analysis:");
    let improvement_over_random = test_accuracy / 0.0625;
    println!("   • {:.1}x better than random guessing", improvement_over_random);
    println!("   • {:.1}% of paper target achieved", (test_accuracy / 0.863) * 100.0);
    
    println!("\n💡 To reach paper performance (86.30%), still needed:");
    println!("   ✗ BERT fine-tuning and integration");
    println!("   ✗ 8-layer Transformer encoder (currently simplified)");
    println!("   ✗ Two-stage training strategy");
    println!("   ✗ GPU acceleration");
    println!("   ✗ Dynamic fusion layer");
    
    println!("\n✓ Successfully implemented:");
    println!("   ✓ 930 psychological features (SEANCE + TAACO + TAALES)");
    println!("   ✓ Pearson correlation feature selection");
    println!("   ✓ Attention-based feature encoding");
    println!("   ✓ Class-balanced training");
    
    println!("\n═══════════════════════════════════════════════════════════════\n");
    println!("🎉 Full PsyAttention implementation complete!");
    println!("\n");

    Ok(())
}

