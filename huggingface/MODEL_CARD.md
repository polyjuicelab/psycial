---
language: en
license: gpl-3.0
tags:
- mbti
- personality-classification
- text-classification
- multi-task-learning
- rust
datasets:
- mbti-kaggle
metrics:
- accuracy
model-index:
- name: psycial-mbti-multitask
  results:
  - task:
      type: text-classification
      name: MBTI Personality Classification
    dataset:
      name: MBTI Kaggle Dataset
      type: mbti-kaggle
    metrics:
    - type: accuracy
      value: 49.80
      name: Overall Test Accuracy
    - type: accuracy
      value: 80.58
      name: E/I Accuracy
    - type: accuracy
      value: 87.15
      name: S/N Accuracy
    - type: accuracy
      value: 81.90
      name: T/F Accuracy
    - type: accuracy
      value: 75.33
      name: J/P Accuracy
---

<div align="center">
  <img src="https://raw.githubusercontent.com/RyanKung/psycial/master/logo.png" alt="Psycial Logo" width="200"/>
  
  # MBTI Personality Classifier - Multi-Task Model
</div>

Production-grade MBTI (Myers-Briggs Type Indicator) personality classification model implemented in Rust with GPU acceleration.

## Model Description

This model predicts MBTI personality type from text using a multi-task learning approach with four independent binary classifiers for each MBTI dimension:
- **E/I**: Extraversion vs Introversion
- **S/N**: Sensing vs Intuition  
- **T/F**: Thinking vs Feeling
- **J/P**: Judging vs Perceiving

### Architecture

- **Input Features**: 5384 dimensions
  - 5000 TF-IDF features (top words from vocabulary)
  - 384 BERT embeddings (sentence-transformers/all-MiniLM-L6-v2)
- **Network**: 4-layer deep MLP
  - Architecture: 5384 → [1024, 768, 512, 256] → 4×2 outputs
  - Dropout: 0.5 (disabled during inference)
  - Optimizer: Adam (lr=0.001)
- **Training**: Per-dimension epochs with weighted loss
  - E/I: 30 epochs, weight=1.2
  - S/N: 30 epochs, weight=1.0
  - T/F: 25 epochs, weight=1.0
  - J/P: 30 epochs, weight=1.3

## Performance

### Overall Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **49.80%** |
| vs Random Baseline (6.25%) | 8.0x better |
| vs TF-IDF Baseline (21.73%) | +129.2% improvement |
| Training Time | ~50 seconds (GPU) |

### Per-Dimension Accuracy

| Dimension | Accuracy | Samples |
|-----------|----------|---------|
| E/I | 80.58% | 1398/1735 |
| S/N | 87.15% | 1512/1735 |
| T/F | 81.90% | 1421/1735 |
| J/P | 75.33% | 1307/1735 |

## Training Data

- **Dataset**: MBTI Kaggle Dataset (8675 samples)
- **Split**: 80% train (6940), 20% test (1735)
- **Classes**: 16 MBTI types (INTJ, ENFP, etc.)
- **Class Distribution**: 
  - I: 77%, E: 23% (imbalanced)
  - N: 86%, S: 14% (highly imbalanced)
  - F: 54%, T: 46% (balanced)
  - J: 60%, P: 40% (moderately imbalanced)

## ⚠️ Important Note

**This model does NOT support Hugging Face Inference API** because it's a custom Rust/PyTorch implementation, not a standard `transformers` model.

To use this model, you have two options:

### Option 1: Use the Web Interface (Easiest)

Visit **https://psycial.0xbase.ai** for instant MBTI predictions - no setup required!

### Option 2: Run Locally (Full Control)

## Usage

### Requirements

- Rust 1.70+
- PyTorch/libtorch (for tch-rs bindings)
- CUDA (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/RyanKung/psycial
cd psycial

# Set up environment
conda create -n psycial python=3.10
conda activate psycial
conda install pytorch

# Build
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
cargo build --release
```

### Download Model from Hugging Face

```python
from huggingface_hub import hf_hub_download

# Download model files
mlp_weights = hf_hub_download(
    repo_id="ElderRyan/psycial",
    filename="mlp_weights_multitask.pt"
)

vectorizer = hf_hub_download(
    repo_id="ElderRyan/psycial",
    filename="tfidf_vectorizer_multitask.json"
)

# Copy to models directory
import shutil
shutil.copy(mlp_weights, "models/mlp_weights_multitask.pt")
shutil.copy(vectorizer, "models/tfidf_vectorizer_multitask.json")
```

### Inference

```bash
# Download model files first (see above)

# Predict single text
./target/release/psycial hybrid predict "I love solving complex problems and thinking deeply about abstract concepts."
```

### Programmatic Usage (Rust)

```rust
use psycial::hybrid::predict::predict_single;

// Load model and predict
predict_single("Your text here")?;
```

### Quick Test

```bash
# Download inference example
wget https://huggingface.co/ElderRyan/polyjuice/raw/main/inference_example.py

# Run it
python inference_example.py
```

## Model Files

This model repository includes:

1. **mlp_weights_multitask.pt** (27MB) - Neural network weights
2. **tfidf_vectorizer_multitask.json** (213KB) - TF-IDF vocabulary and IDF weights  
3. **feature_normalizer.json** (5KB) - Feature normalization parameters (optional)
4. **feature_selector.json** (1KB) - Pearson feature selector indices (optional)

## Limitations

1. **Moderate Accuracy**: 49.80% is significantly better than random/baseline but still has room for improvement
2. **Class Imbalance**: Model may favor majority classes (I, N, F, J)
3. **Data Bias**: Trained on online forum posts, may not generalize to all text types
4. **Language**: English only
5. **MBTI Validity**: MBTI itself has limited scientific validity

## Ethical Considerations

- MBTI is not scientifically validated for hiring/clinical decisions
- Predictions should be used for entertainment/research only
- Be aware of class imbalances and potential biases
- Do not use for discriminatory purposes

## Technical Details

### Framework
- **Language**: Rust
- **ML Framework**: tch-rs (PyTorch bindings)
- **BERT Model**: sentence-transformers/all-MiniLM-L6-v2
- **Device**: CUDA-capable GPU (falls back to CPU)

### Key Innovations

1. **Dropout Bug Fix**: Discovered and fixed critical bug where dropout remained active during inference, causing ~5% accuracy loss
2. **Per-Dimension Optimization**: Different epochs and loss weights for each MBTI dimension
3. **Multi-Task Learning**: Four independent binary classifiers instead of 16-way classification

## Citation

If you use this model, please cite:

```bibtex
@software{psycial_mbti_2025,
  title = {Psycial: Multi-Task MBTI Personality Classifier},
  author = {ElderRyan},
  year = {2025},
  url = {https://huggingface.co/ElderRyan/psycial},
  github = {https://github.com/RyanKung/psycial}
}
```

## License

GNU General Public License v3.0 (GPLv3) - See LICENSE file for details

## Acknowledgments

- **rust-bert**: Guillaume BE for BERT Rust implementation
- **Dataset**: MBTI Kaggle Dataset
- **sentence-transformers**: all-MiniLM-L6-v2 model

