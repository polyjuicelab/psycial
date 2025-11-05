//! Command-line interface for the hybrid MBTI classifier.

use super::predict::predict_single;
use super::train::train_model;
use std::error::Error;

/// Print command-line usage information.
pub fn print_usage() {
    println!("Usage:");
    println!("  cargo run --release --features bert --bin psycial -- hybrid [COMMAND] [OPTIONS]\n");
    println!("Commands:");
    println!("  train              Train new GPU model (saves to models/)");
    println!("  predict TEXT       Predict single text (requires trained model)");
    println!("  help               Show this help\n");
    println!("Options:");
    println!(
        "  --multi-task       Use multi-task model (4 binary classifiers: E/I, S/N, T/F, J/P)"
    );
    println!("  --single-task      Use single-task model (16-way classification)");
    println!("                     Default: uses config.toml setting\n");
    println!("Examples:");
    println!("  ./target/release/psycial hybrid train --multi-task");
    println!("  ./target/release/psycial hybrid train --single-task");
    println!("  ./target/release/psycial hybrid predict \"I love solving problems\"");
}

/// Main entry point for the hybrid classifier CLI.
///
/// # Arguments
///
/// * `args` - Command-line arguments (including program name)
pub fn main_hybrid(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let command = if args.len() > 1 {
        args[1].as_str()
    } else {
        "train"
    };

    match command {
        "train" => {
            // Check for model type flags
            let model_type_override = if args.contains(&"--multi-task".to_string()) {
                Some("multitask")
            } else if args.contains(&"--single-task".to_string()) {
                Some("single")
            } else {
                None
            };
            train_model(model_type_override)
        }
        "predict" => {
            if args.len() < 3 {
                println!("Error: TEXT argument required\n");
                print_usage();
                return Ok(());
            }
            predict_single(&args[2])
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        _ => {
            println!("Unknown command: {}\n", command);
            print_usage();
            Ok(())
        }
    }
}
