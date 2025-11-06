<div align="center">
  <img src="logo.png" alt="Psycial Logo" width="200"/>
  
  # MBTI Personality Classifier

  [![Rust](https://img.shields.io/badge/Rust-100%25-orange)](https://www.rust-lang.org/)
  [![Accuracy](https://img.shields.io/badge/Best-49.80%25-success)](https://github.com)
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

  **Production-grade MBTI personality classification system** with neural network implementation in pure Rust.
</div>

---

## Performance Summary

| Model | Method | Accuracy | vs Random | Training Time |
|-------|--------|----------|-----------|---------------|
| **V7** | **🏆 TF-IDF + BERT + Multi-Task GPU** | **49.80%** | **8.0x** | ~50s | ⭐ **BEST** |
| V6 | BERT + MLP (single-task) | 31.99% | 5.1x | 322s |
| V1 | TF-IDF + Naive Bayes | 21.73% | 3.5x | 2s |
| V2 | 9 Psychological Features | 21.21% | 3.4x | 3s |
| V3 | 930 Psychological Features | 20.12% | 3.2x | 30s |
| V5 | BERT Only + Cosine | 18.39% | 2.9x | 583s |

**Random Baseline**: 6.25% (16 classes)

### V7 Multi-Task Model Breakdown
| Dimension | Accuracy | Notes |
|-----------|----------|-------|
| E/I | 80.58% | Extraversion vs Introversion |
| S/N | 87.15% | Sensing vs Intuition (best) |
| T/F | 81.90% | Thinking vs Feeling |
| J/P | 75.33% | Judging vs Perceiving |

---

## Quick Start

### As a CLI Tool

```bash
# Show all available models
cargo run --release

# Run baseline model
cargo run --release -- baseline

# Run best model (multi-task hybrid)
cargo run --release -- hybrid train --multi-task

# Run BERT model
cargo run --release -- bert-mlp
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for detailed CLI usage.

### As a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
psycial = { git = "https://github.com/yourusername/psycial", features = ["bert"] }
```

Use in your code:

```rust
use psycial::{load_data, MultiTaskGpuMLP, RustBertEncoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data and initialize BERT
    let records = load_data("data/mbti_1.csv")?;
    let bert = RustBertEncoder::new()?;
    
    // Extract features
    let texts: Vec<String> = records.iter().map(|r| r.posts.clone()).collect();
    let labels: Vec<String> = records.iter().map(|r| r.mbti_type.clone()).collect();
    let features = bert.extract_features_batch(&texts)?;
    
    // Train multi-task model (4 binary classifiers: E/I, S/N, T/F, J/P)
    let mut model = MultiTaskGpuMLP::new(384, vec![256, 128], 0.001, 0.5);
    model.train(&features, &labels, 20, 32);
    
    // Predict
    let test_features = bert.extract_features("I love planning everything.")?;
    let mbti_type = model.predict(&test_features);
    println!("Predicted: {}", mbti_type);
    
    Ok(())
}
```

See [LIBRARY_USAGE.md](LIBRARY_USAGE.md) for detailed library integration examples.

---

## Key Technical Achievements

### 1. Neural Network Beats Statistics

```
Statistical (TF-IDF + NB):    21.73%
Deep Learning (BERT + MLP):   31.99%  (+47% improvement)
```

**Proven**: Deep learning can leverage BERT features that simple classifiers cannot.

### 2. Pure Rust Implementation

- ✅ MLP with backpropagation
- ✅ BERT integration (rust-bert)
- ✅ 930 psychological features
- ✅ Automatic feature engineering
- ✅ No Python dependencies for inference

### 3. Production Ready

- Clean Rust API
- Comprehensive error handling
- Modular architecture
- Full test coverage
- Metal GPU support (M1/M2/M3)

---

## Architecture

### BERT + MLP (Best Performing)

```
Text Input
    ↓
BERT Encoder (384-dim embeddings)
    ↓
MLP Neural Network (384 -> 256 -> 128 -> 16)
    ├─ Xavier initialization
    ├─ ReLU activation
    ├─ Softmax output
    └─ SGD optimizer
    ↓
MBTI Type (16 classes)
```

### Features

- **BERT Model**: all-MiniLM-L12-v2 (sentence-transformers)
- **Embedding Dim**: 384
- **MLP Architecture**: 3 layers (256, 128 hidden units)
- **Activation**: ReLU
- **Optimizer**: SGD with learning rate 0.001
- **Training**: 25 epochs, batch size 32

---

## Installation

### Requirements

- Rust 1.70+
- 8GB RAM minimum
- 2GB disk space (for models)

### Setup

```bash
# Clone
git clone <repo>
cd snapMBTI

# Build (downloads libtorch automatically on first build)
cargo build --release --features bert --bin bert-mlp

# This will take ~60 minutes first time (downloading & compiling libtorch)
# Subsequent builds are fast
```

---

## Usage

### Run Best Model

```bash
cargo run --release --features bert --bin bert-mlp
```

### Run All Models

```bash
# V1: TF-IDF baseline
cargo run --release --bin baseline

# V2-V3: Psychological features
cargo run --release --bin psyattention
cargo run --release --bin psyattention-full

# V5: BERT experiments
cargo run --release --features bert --bin bert-only

# V6: BERT + MLP (best)
cargo run --release --features bert --bin bert-mlp
```

---

## Technical Details

### Dataset

- **Source**: MBTI Kaggle Dataset
- **Size**: 8,675 samples
- **Split**: 80% train (6,940) / 20% test (1,735)
- **Classes**: 16 MBTI types
- **Imbalance**: INFP (21%), ENTP (~1%)

### MLP Implementation

**Pure Rust** neural network with:
- Xavier weight initialization  
- Backpropagation
- Mini-batch SGD
- ReLU activations
- Softmax output
- Cross-entropy loss

**Code**: `src/neural_net.rs` (259 lines)

### BERT Integration

**Library**: [rust-bert](https://github.com/guillaume-be/rust-bert) v0.22
- Automatic model downloads
- Sentence embeddings
- Metal GPU support
- Pure Rust API (libtorch backend)

**Code**: `src/psyattention/bert_rustbert.rs`

---

## Project Structure

```
snapMBTI/
├── src/
│   ├── main.rs                  # V1: Baseline
│   ├── bert_mlp.rs              # V6: BERT + MLP (BEST)
│   ├── bert_only.rs             # V5: BERT experiments
│   ├── neural_net.rs            # MLP implementation
│   └── psyattention/            # Psychological features
│       ├── bert_rustbert.rs     # BERT encoder
│       ├── seance.rs            # 271 emotion features
│       ├── taaco.rs             # 168 coherence features
│       ├── taales.rs            # 491 complexity features
│       └── ...
├── data/mbti_1.csv              # Dataset
└── docs/                        # Documentation
```

---

## Performance Analysis

### Why Neural Network Works

**BERT + k-NN**: 18.39%
- High-dimensional features
- Simple nearest-neighbor
- Cannot learn complex patterns
- **Result**: Underutilizes BERT

**BERT + MLP**: 31.99%
- High-dimensional features
- Neural network classifier
- Learns non-linear decision boundaries
- **Result**: Properly utilizes BERT

### Comparison to Paper

| Component | Our Implementation | Paper | Gap |
|-----------|-------------------|-------|-----|
| BERT | ✅ sentence-transformers | ✅ BERT-base | Similar |
| Classifier | ✅ 3-layer MLP | ✅ 8-layer Transformer + MLP | Architecture depth |
| Training | ✅ Single-stage | ✅ Two-stage (BERT alignment) | Training strategy |
| Accuracy | **31.99%** | **86.30%** | 54.31% |

**To reach 86%**: Need 8-layer Transformer encoder + two-stage training

---

## Benchmarks (M1 Max)

| Model | Compilation | Training | Inference | Accuracy |
|-------|------------|----------|-----------|----------|
| Baseline | 30s | 2s | <1s | 21.73% |
| BERT + MLP | 60min (first) | 322s | 61s | 31.99% |

*First compilation downloads libtorch (~500MB)

---

## Dependencies

```toml
[dependencies]
csv = "1.3"
serde = "1.0"
rand = "0.8"
ndarray = "0.16"

# Optional: BERT support
rust-bert = { version = "0.22", optional = true, features = ["download-libtorch"] }
```

---

## Citation

```bibtex
@software{snapMBTI2025,
  title={MBTI Personality Classifier with Neural Networks},
  author={Ryan Kung},
  year={2025},
  note={Pure Rust, 31.99\% accuracy, BERT + MLP implementation}
}
```

---

## License

MIT

---

## Acknowledgments

- **rust-bert**: Guillaume BE
- **Dataset**: MBTI Kaggle Dataset

---

**Status**: Production Ready  
**Best Model**: BERT + MLP (31.99%)  
**Platform**: macOS (Metal GPU), Linux, Windows  
**Language**: 100% Rust
