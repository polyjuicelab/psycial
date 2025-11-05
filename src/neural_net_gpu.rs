use std::collections::HashMap;
use std::error::Error;
/// GPU-accelerated MLP using tch (PyTorch bindings)
/// Much faster than CPU implementation on CUDA devices
#[cfg(feature = "bert")]
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[cfg(feature = "bert")]
pub struct GpuMLP {
    device: Device,
    vs: nn::VarStore,
    net: nn::Sequential,
    class_to_idx: HashMap<String, usize>,
    idx_to_class: HashMap<usize, String>,
    learning_rate: f64,
}

#[cfg(feature = "bert")]
impl GpuMLP {
    pub fn new(
        input_dim: i64,
        hidden_dims: Vec<i64>,
        output_dim: i64,
        learning_rate: f64,
        dropout_rate: f64,
    ) -> Self {
        // Auto-detect GPU
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        println!("🎮 MLP Device: {:?}", device);

        let mut vs = nn::VarStore::new(device);
        vs.set_kind(tch::Kind::Float); // Use f32 to match input data type
        let root = vs.root();

        // Build network
        let mut net = nn::seq();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            net = net
                .add(nn::linear(
                    &root / format!("fc{}", i),
                    current_dim,
                    hidden_dim,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add_fn(move |x| x.dropout(dropout_rate, true));
            current_dim = hidden_dim;
        }

        // Output layer
        net = net.add(nn::linear(
            &root / "output",
            current_dim,
            output_dim,
            Default::default(),
        ));

        GpuMLP {
            device,
            vs,
            net,
            class_to_idx: HashMap::new(),
            idx_to_class: HashMap::new(),
            learning_rate,
        }
    }

    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        labels: &[String],
        epochs: i64,
        batch_size: i64,
    ) {
        println!("🏋️  Training on {:?}...", self.device);

        // Map labels to indices
        let unique_labels: Vec<String> = {
            let mut labels_set: Vec<String> = labels.to_vec();
            labels_set.sort();
            labels_set.dedup();
            labels_set
        };

        for (i, label) in unique_labels.iter().enumerate() {
            self.class_to_idx.insert(label.clone(), i);
            self.idx_to_class.insert(i, label.clone());
        }

        // Convert to tensors
        let n_samples = features.len() as i64;
        let feature_dim = features[0].len() as i64;

        // Flatten features to 1D vec
        let flat_features: Vec<f32> = features
            .iter()
            .flat_map(|f| f.iter().map(|&x| x as f32))
            .collect();

        let x_train = Tensor::from_slice(&flat_features)
            .view([n_samples, feature_dim])
            .to(self.device);

        // Convert labels to tensor
        let label_indices: Vec<i64> = labels
            .iter()
            .map(|l| *self.class_to_idx.get(l).unwrap() as i64)
            .collect();

        let y_train = Tensor::from_slice(&label_indices).to(self.device);

        // Optimizer
        let mut opt = nn::Adam::default()
            .build(&self.vs, self.learning_rate)
            .unwrap();

        // Training loop
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;
            let mut batches = 0;

            // Shuffle indices
            let indices = Tensor::randperm(n_samples, (tch::Kind::Int64, self.device));

            for batch_start in (0..n_samples).step_by(batch_size as usize) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_indices = indices.narrow(0, batch_start, batch_end - batch_start);

                // Get batch
                let batch_x = x_train.index_select(0, &batch_indices);
                let batch_y = y_train.index_select(0, &batch_indices);

                // Forward
                let logits = self.net.forward(&batch_x);
                let loss = logits.cross_entropy_for_logits(&batch_y);

                // Backward
                opt.backward_step(&loss);

                // Stats
                total_loss += f64::try_from(&loss).unwrap();
                let predictions = logits.argmax(-1, false);
                correct +=
                    i64::try_from(predictions.eq_tensor(&batch_y).sum(tch::Kind::Int64)).unwrap();
                batches += 1;
            }

            if (epoch + 1) % 5 == 0 {
                let avg_loss = total_loss / batches as f64;
                let accuracy = correct as f64 / n_samples as f64 * 100.0;
                println!(
                    "  Epoch {:3}/{}: Loss={:.4}, Acc={:.2}%",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    accuracy
                );
            }
        }

        println!("✅ Training complete!");
    }

    pub fn predict(&self, features: &[f64]) -> String {
        let feature_tensor = Tensor::from_slice(features)
            .view([1, features.len() as i64])
            .to_kind(tch::Kind::Float)
            .to(self.device);

        tch::no_grad(|| {
            let logits = self.net.forward(&feature_tensor);
            let pred_idx = i64::try_from(logits.argmax(-1, false)).unwrap() as usize;

            self.idx_to_class
                .get(&pred_idx)
                .cloned()
                .unwrap_or_else(|| "UNKNOWN".to_string())
        })
    }

    /// Batch prediction
    pub fn predict_batch(&self, features_batch: &[Vec<f64>]) -> Vec<String> {
        features_batch
            .iter()
            .map(|features| self.predict(features))
            .collect()
    }

    #[allow(dead_code)]
    pub fn num_layers(&self) -> usize {
        // Approximate based on hidden layers
        3 // Will be correct based on architecture
    }

    /// Save model weights to file
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        println!("  ✓ MLP weights saved to {}", path);
        Ok(())
    }

    /// Load model weights from file
    pub fn load(
        path: &str,
        input_dim: i64,
        hidden_dims: Vec<i64>,
        output_dim: i64,
        dropout_rate: f64,
    ) -> Result<Self, Box<dyn Error>> {
        // Create model structure
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let mut vs = nn::VarStore::new(device);
        vs.set_kind(tch::Kind::Float); // Use f32 to match input data type
        let root = vs.root();

        // Rebuild network structure
        let mut net = nn::seq();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            net = net
                .add(nn::linear(
                    &root / format!("fc{}", i),
                    current_dim,
                    hidden_dim,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add_fn(move |x| x.dropout(dropout_rate, false)); // Inference mode
            current_dim = hidden_dim;
        }

        net = net.add(nn::linear(
            &root / "output",
            current_dim,
            output_dim,
            Default::default(),
        ));

        // Load weights
        vs.load(path)?;

        Ok(GpuMLP {
            device,
            vs,
            net,
            class_to_idx: HashMap::new(),
            idx_to_class: HashMap::new(),
            learning_rate: 0.001, // Default for loaded models
        })
    }

    /// Save class mapping
    pub fn save_class_mapping(&self, path: &str) -> Result<(), Box<dyn Error>> {
        use serde_json;
        let mapping = serde_json::json!({
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
        });
        std::fs::write(path, serde_json::to_string_pretty(&mapping)?)?;
        Ok(())
    }

    /// Load class mapping
    pub fn load_class_mapping(&mut self, path: &str) -> Result<(), Box<dyn Error>> {
        use serde_json;
        let json = std::fs::read_to_string(path)?;
        let mapping: serde_json::Value = serde_json::from_str(&json)?;

        if let Some(c2i) = mapping["class_to_idx"].as_object() {
            for (k, v) in c2i {
                if let Some(idx) = v.as_u64() {
                    self.class_to_idx.insert(k.clone(), idx as usize);
                }
            }
        }

        if let Some(i2c) = mapping["idx_to_class"].as_object() {
            for (k, v) in i2c {
                if let (Ok(idx), Some(class)) = (k.parse::<usize>(), v.as_str()) {
                    self.idx_to_class.insert(idx, class.to_string());
                }
            }
        }

        Ok(())
    }
}

// Fallback for non-BERT builds
#[cfg(not(feature = "bert"))]
pub struct GpuMLP;

#[cfg(not(feature = "bert"))]
impl GpuMLP {
    pub fn new(_: i64, _: Vec<i64>, _: i64, _: f64, _: f64) -> Self {
        GpuMLP
    }

    pub fn train(&mut self, _: &[Vec<f64>], _: &[String], _: i64, _: i64) {
        eprintln!("GPU MLP requires bert feature");
    }

    pub fn predict(&self, _: &[f64]) -> String {
        "UNKNOWN".to_string()
    }

    pub fn num_layers(&self) -> usize {
        0
    }
}
