# Examples

This directory contains examples of how to use Psycial as a library.

## Running Examples

```bash
# Basic usage
cargo run --example basic_usage

# BERT classifier (requires BERT feature)
cargo run --example bert_usage --features bert
```

## Available Examples

### `basic_usage.rs`
Demonstrates:
- Loading data
- Training a classifier
- Making predictions
- Saving and loading models

**Dependencies**: None (uses baseline model)

### `bert_usage.rs`
Demonstrates:
- Using BERT-based classifier
- Higher accuracy predictions
- GPU acceleration (if available)

**Dependencies**: Requires `bert` feature and libtorch

## Creating Your Own

Use these examples as templates for your own projects. Simply copy the relevant code and adapt it to your needs.

### Template Structure

```rust
use psycial::{load_data, BaselineClassifier};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // 1. Load data
    let records = load_data("data/mbti_1.csv")?;
    
    // 2. Train classifier
    let mut classifier = BaselineClassifier::new();
    classifier.train(&records)?;
    
    // 3. Make predictions
    let prediction = classifier.predict("your text")?;
    println!("Prediction: {}", prediction);
    
    Ok(())
}
```

## Integration Examples

See [INTEGRATION.md](../INTEGRATION.md) for examples of:
- Web API servers
- CLI integration
- Batch processing
- Discord bots
- And more!

## Need Help?

- 📖 [Library Usage Guide](../LIBRARY_USAGE.md)
- 🚀 [Integration Guide](../INTEGRATION.md)
- 💬 Open an issue on GitHub

