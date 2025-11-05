# Quick Reference Guide

## Training

### Multi-Task Model (Recommended) 🌟

```bash
# Using command-line flag
./target/release/psycial hybrid train --multi-task

# Expected output:
# - models/tfidf_vectorizer_multitask.json
# - models/mlp_weights_multitask.pt
# - Accuracy: ~55-60% (expected)
```

### Single-Task Model

```bash
# Using command-line flag
./target/release/psycial hybrid train --single-task

# Expected output:
# - models/tfidf_vectorizer_single.json
# - models/mlp_weights_single.pt
# - models/class_mapping_single.json
# - Accuracy: ~49%
```

### Default (from config.toml)

```bash
./target/release/psycial hybrid train
# Uses model_type from config.toml
```

## Prediction

```bash
./target/release/psycial hybrid predict "I love organizing and planning everything"
# Output: Predicted MBTI Type: ISTJ
```

## Configuration Priority

```
Command-line flags  >  config.toml  >  Hard-coded defaults
```

## Model Type Comparison

| Feature | Multi-Task | Single-Task |
|---------|------------|-------------|
| **Architecture** | 4 binary classifiers | 16-way classifier |
| **Theory** | ✅ MBTI dimensions | Data-driven |
| **Expected Accuracy** | **55-60%** | 49% |
| **Interpretability** | High | Low |
| **Training Speed** | Similar | Similar |
| **File Count** | 2 files | 3 files |

## Common Commands

```bash
# Build
cargo build --release

# Train multi-task (recommended)
./target/release/psycial hybrid train --multi-task

# Train single-task (for comparison)
./target/release/psycial hybrid train --single-task

# Predict
./target/release/psycial hybrid predict "your text"

# Help
./target/release/psycial hybrid help

# Run example
cargo run --release --example simple_usage
```

## Customizing Hyperparameters

Edit `config.toml`:

```toml
[model]
hidden_layers = [512, 256]  # Smaller network
learning_rate = 0.0005      # Lower learning rate
dropout_rate = 0.6          # Stronger regularization

[training]
epochs = 30
batch_size = 128
```

Then train:

```bash
./target/release/psycial hybrid train --multi-task
```

## Checking GPU

```bash
# The program will automatically show GPU info:
# 🔍 CUDA Diagnostics:
#    CUDA available: true
#    CUDA device count: 1
#    Using GPU device: 0
```

## Model Files

After training both types:

```
models/
├── tfidf_vectorizer_multitask.json
├── mlp_weights_multitask.pt
├── tfidf_vectorizer_single.json
├── mlp_weights_single.pt
└── class_mapping_single.json
```

## Troubleshooting

**No GPU detected?**
```bash
# Check CUDA
nvidia-smi

# Check PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

**Model files conflict?**
- Don't worry! Multi-task and single-task use different filenames automatically.

**Config file not found?**
- It's optional. The program uses defaults if `config.toml` is missing.

---

🎯 **Recommended Workflow**:
1. `./target/release/psycial hybrid train --multi-task`
2. Check accuracy
3. Adjust `config.toml` if needed
4. Retrain and compare

