//! Model saving functionality for trained models.

use super::config::OutputConfig;
use super::tfidf::TfidfVectorizer;
use crate::neural_net_gpu::GpuMLP;
use crate::neural_net_gpu_multitask::MultiTaskGpuMLP;
use std::error::Error;

/// Save a multi-task model to disk.
///
/// This saves the TF-IDF vectorizer and MLP weights with a "_multitask" suffix
/// to avoid conflicts with single-task models.
///
/// # Arguments
///
/// * `mlp` - The trained multi-task MLP model
/// * `tfidf` - The fitted TF-IDF vectorizer
/// * `output` - Output configuration specifying paths
pub fn save_model(
    mlp: &MultiTaskGpuMLP,
    tfidf: &TfidfVectorizer,
    output: &OutputConfig,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&output.model_dir)?;

    println!("Saving model...");

    // Add suffix for multi-task models
    let tfidf_file = output.tfidf_file.replace(".json", "_multitask.json");
    let mlp_file = output.mlp_file.replace(".pt", "_multitask.pt");

    let tfidf_path = format!("{}/{}", output.model_dir, tfidf_file);
    let mlp_path = format!("{}/{}", output.model_dir, mlp_file);

    tfidf.save(&tfidf_path)?;
    mlp.save(&mlp_path)?;

    println!("\n✓ Multi-Task Model saved:");
    println!("  - {}", tfidf_path);
    println!("  - {}", mlp_path);

    println!("\n===================================================================\n");
    println!("Training complete!\n");
    println!("Note: Multi-task model uses different files to avoid conflicts.");
    println!("To predict: ./target/release/psycial hybrid predict \"your text here\"\n");

    Ok(())
}

/// Save a single-task model to disk.
///
/// This saves the TF-IDF vectorizer, MLP weights, and class mapping with a
/// "_single" suffix to avoid conflicts with multi-task models.
///
/// # Arguments
///
/// * `mlp` - The trained single-task MLP model
/// * `tfidf` - The fitted TF-IDF vectorizer
/// * `output` - Output configuration specifying paths
pub fn save_model_singletask(
    mlp: &GpuMLP,
    tfidf: &TfidfVectorizer,
    output: &OutputConfig,
) -> Result<(), Box<dyn Error>> {
    std::fs::create_dir_all(&output.model_dir)?;

    println!("Saving model...");

    // Add suffix for single-task models
    let tfidf_file = output.tfidf_file.replace(".json", "_single.json");
    let mlp_file = output.mlp_file.replace(".pt", "_single.pt");
    let class_file = output.class_mapping_file.replace(".json", "_single.json");

    let tfidf_path = format!("{}/{}", output.model_dir, tfidf_file);
    let mlp_path = format!("{}/{}", output.model_dir, mlp_file);
    let class_path = format!("{}/{}", output.model_dir, class_file);

    tfidf.save(&tfidf_path)?;
    mlp.save(&mlp_path)?;
    mlp.save_class_mapping(&class_path)?;

    println!("\n✓ Single-Task Model saved:");
    println!("  - {}", tfidf_path);
    println!("  - {}", mlp_path);
    println!("  - {}", class_path);

    println!("\n===================================================================\n");
    println!("Training complete!\n");
    println!("Note: Single-task model uses different files to avoid conflicts.");
    println!("To predict: ./target/release/psycial hybrid predict \"your text here\"\n");

    Ok(())
}
