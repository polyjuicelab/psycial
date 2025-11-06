# Upload Instructions for ElderRyan/psycial

## Prerequisites

1. Install huggingface_hub:
```bash
pip install huggingface_hub
```

2. Login to Hugging Face:
```bash
huggingface-cli login
# Enter your token when prompted
```

## Step-by-Step Upload

### Step 1: Train the Best Model

```bash
cd /home/ryan/dev/farcaster/psycial

conda activate psycial
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"

# Train multi-task model (best performance: 49.80%)
./target/release/psycial hybrid train --multi-task
```

### Step 2: Verify Model Files

```bash
./huggingface/test_upload.sh
```

Expected output:
```
✓ mlp_weights_multitask.pt (27M)
✓ tfidf_vectorizer_multitask.json (213K)
```

### Step 3: Upload to Hugging Face

```bash
cd huggingface
python upload_to_hf.py --repo-id ElderRyan/psycial
```

## What Gets Uploaded

1. **mlp_weights_multitask.pt** - Neural network weights (~27MB)
2. **tfidf_vectorizer_multitask.json** - TF-IDF vocabulary (~213KB)
3. **README.md** - Auto-generated from MODEL_CARD.md
4. **config.json** - Model metadata

## Access Your Model

After upload, your model will be available at:
- **URL**: https://huggingface.co/ElderRyan/psycial
- **Downloads**: Via `huggingface_hub` library

## Download Example

```python
from huggingface_hub import hf_hub_download

weights = hf_hub_download(
    repo_id="ElderRyan/psycial",
    filename="mlp_weights_multitask.pt"
)

vectorizer = hf_hub_download(
    repo_id="ElderRyan/psycial", 
    filename="tfidf_vectorizer_multitask.json"
)
```

## Troubleshooting

### Authentication Error
```bash
# Re-login
huggingface-cli login --token YOUR_TOKEN
```

### Model Not Found
Train the model first:
```bash
./target/release/psycial hybrid train --multi-task
```

### Permission Denied
Make sure you have write access to `ElderRyan/psycial` repository.
