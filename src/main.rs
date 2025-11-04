/// Unified CLI for MBTI Personality Prediction
/// All models accessible via subcommands
use std::env;
use std::process;

mod baseline;
mod bert_mlp;
mod bert_only;
mod hybrid_tfidf_bert;
mod neural_net;
mod neural_net_advanced;
mod neural_net_gpu;
mod psyattention;
mod psyattention_candle;
mod psyattention_full;

fn print_help() {
    println!(
        "
╔══════════════════════════════════════════════════════════════════╗
║          MBTI Personality Prediction - Unified CLI              ║
╚══════════════════════════════════════════════════════════════════╝

USAGE:
    psycial <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    baseline              TF-IDF + Naive Bayes baseline model
    psyattention          PsyAttention with 9 features
    psyattention-full     PsyAttention with 930→108 features
    psyattention-bert     PsyAttention with BERT integration
    bert-only             BERT-only classification
    bert-mlp              BERT + MLP hybrid model
    hybrid                TF-IDF + BERT hybrid (interactive)
    help, --help, -h      Show this help message

EXAMPLES:
    # Train and test baseline model
    psycial baseline

    # Train PsyAttention model
    psycial psyattention

    # Full PsyAttention with feature selection
    psycial psyattention-full

    # BERT models (all features enabled by default)
    psycial bert-only
    psycial bert-mlp
    psycial psyattention-bert

    # Interactive hybrid model
    psycial hybrid

OPTIONS:
    Each subcommand has its own options. Use:
        psycial <SUBCOMMAND> --help

NOTES:
    - BERT features are enabled by default
    - To build without BERT: cargo build --no-default-features
"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(1);
    }

    let subcommand = &args[1];

    // Pass remaining args to subcommand
    let sub_args: Vec<String> = args.iter().skip(1).cloned().collect();

    let result = match subcommand.as_str() {
        "baseline" => baseline::main_baseline(sub_args),
        "psyattention" => psyattention::main_psyattention(sub_args),
        "psyattention-full" => psyattention_full::main_psyattention_full(sub_args),
        "psyattention-bert" => psyattention_candle::main_psyattention_bert(sub_args),
        "bert-only" => bert_only::main_bert_only(sub_args),
        "bert-mlp" => bert_mlp::main_bert_mlp(sub_args),
        "hybrid" => hybrid_tfidf_bert::main_hybrid(sub_args),
        "help" | "--help" | "-h" => {
            print_help();
            process::exit(0);
        }
        _ => {
            eprintln!("Error: Unknown subcommand '{}'", subcommand);
            eprintln!("\nRun 'psycial help' for usage information.");
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
