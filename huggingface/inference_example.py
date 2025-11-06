#!/usr/bin/env python3
"""
Inference example for Polyjuice MBTI model from Hugging Face

Since this is a custom Rust/PyTorch model, it cannot use HF Inference API.
Users need to download the model files and use the Rust binary for inference.

This script shows how to download and prepare for inference.
"""

from huggingface_hub import hf_hub_download
import os
import subprocess
import json

def download_model(repo_id="ElderRyan/psycial", cache_dir="./model_cache"):
    """Download model files from Hugging Face"""
    
    print(f"Downloading model from {repo_id}...")
    
    files_to_download = [
        "mlp_weights_multitask.pt",
        "tfidf_vectorizer_multitask.json",
        "config.json"
    ]
    
    downloaded_paths = {}
    
    for filename in files_to_download:
        print(f"  Downloading {filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir
        )
        downloaded_paths[filename] = path
        print(f"    ✓ Saved to: {path}")
    
    return downloaded_paths

def show_usage():
    """Show how to use the downloaded model"""
    
    print("\n" + "="*60)
    print("MODEL DOWNLOADED SUCCESSFULLY")
    print("="*60)
    print("\nThis is a Rust-based model. To use it:")
    print("\n1. Clone the Rust project:")
    print("   git clone https://github.com/RyanKung/psycial")
    print("   cd psycial")
    print("\n2. Copy downloaded model files:")
    print("   mkdir -p models")
    print("   cp <downloaded_path>/mlp_weights_multitask.pt models/")
    print("   cp <downloaded_path>/tfidf_vectorizer_multitask.json models/")
    print("\n3. Build and run:")
    print("   cargo build --release")
    print("   ./target/release/psycial hybrid predict \"Your text here\"")
    print("\n" + "="*60)
    print("\nAlternatively, use the web interface at:")
    print("https://psycial.0xbase.ai")
    print("="*60 + "\n")

def main():
    print("\n╔═══════════════════════════════════════════════════════════╗")
    print("║  Psycial MBTI Classifier - Model Download                ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    # Download model
    paths = download_model()
    
    # Load and display config
    with open(paths["config.json"], 'r') as f:
        config = json.load(f)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Model Type: {config.get('model_type', 'N/A')}")
    print(f"Input Features: {config.get('input_features', 'N/A')}")
    print(f"Architecture: {config.get('architecture', 'N/A')}")
    print(f"\nAccuracy:")
    acc = config.get('accuracy', {})
    print(f"  Overall: {acc.get('overall', 'N/A')}%")
    print(f"  E/I: {acc.get('e_i', 'N/A')}%")
    print(f"  S/N: {acc.get('s_n', 'N/A')}%")
    print(f"  T/F: {acc.get('t_f', 'N/A')}%")
    print(f"  J/P: {acc.get('j_p', 'N/A')}%")
    
    show_usage()

if __name__ == "__main__":
    main()

