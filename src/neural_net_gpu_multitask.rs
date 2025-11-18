/// Multi-task GPU-accelerated MLP for MBTI 4-dimension prediction
/// Each dimension (E/I, S/N, T/F, J/P) is predicted independently
use std::error::Error;

#[cfg(feature = "bert")]
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[cfg(feature = "bert")]
pub struct MultiTaskGpuMLP {
    device: Device,
    vs: nn::VarStore,

    // Shared feature extraction layers (list of linear layers)
    shared_layers: Vec<nn::Linear>,

    // Four independent output heads for each dimension
    head_ei: nn::Linear, // Extraversion/Introversion
    head_sn: nn::Linear, // Sensing/Intuition
    head_tf: nn::Linear, // Thinking/Feeling
    head_jp: nn::Linear, // Judging/Perceiving

    learning_rate: f64,
    weight_decay: f64,
    dropout_rate: f64,
}

#[cfg(feature = "bert")]
impl MultiTaskGpuMLP {
    pub fn new(
        input_dim: i64,
        shared_layers: Vec<i64>,
        learning_rate: f64,
        dropout_rate: f64,
        weight_decay: f64,
    ) -> Self {
        // Auto-detect GPU
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        println!("🎮 Multi-Task MLP Device: {:?}", device);

        let mut vs = nn::VarStore::new(device);
        vs.set_kind(tch::Kind::Float); // Use f32 to match input data type
        let root = vs.root();

        // Build shared network layers (list of linear layers)
        let mut shared_linear_layers = Vec::new();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in shared_layers.iter().enumerate() {
            let linear = nn::linear(
                &root / format!("shared_fc{}", i),
                current_dim,
                hidden_dim,
                Default::default(),
            );
            shared_linear_layers.push(linear);
            current_dim = hidden_dim;
        }

        // Four independent binary classification heads
        let head_ei = nn::linear(&root / "head_ei", current_dim, 2, Default::default());
        let head_sn = nn::linear(&root / "head_sn", current_dim, 2, Default::default());
        let head_tf = nn::linear(&root / "head_tf", current_dim, 2, Default::default());
        let head_jp = nn::linear(&root / "head_jp", current_dim, 2, Default::default());

        MultiTaskGpuMLP {
            device,
            vs,
            shared_layers: shared_linear_layers,
            head_ei,
            head_sn,
            head_tf,
            head_jp,
            learning_rate,
            weight_decay,
            dropout_rate,
        }
    }

    /// Train the multi-task model with uniform epochs for all dimensions
    #[allow(dead_code)]
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        labels: &[String], // MBTI types like "INTJ", "ENFP"
        epochs: i64,
        batch_size: i64,
    ) {
        // Use uniform epochs for all dimensions
        self.train_per_dimension(features, labels, vec![epochs; 4], batch_size);
    }

    /// Train the multi-task model with per-dimension epochs (uniform weights)
    /// per_dimension_epochs: [E/I, S/N, T/F, J/P]
    pub fn train_per_dimension(
        &mut self,
        features: &[Vec<f64>],
        labels: &[String], // MBTI types like "INTJ", "ENFP"
        per_dimension_epochs: Vec<i64>,
        batch_size: i64,
    ) {
        // Use uniform weights
        self.train_per_dimension_weighted(
            features,
            labels,
            per_dimension_epochs,
            vec![1.0; 4],
            batch_size,
        );
    }

    /// Train the multi-task model with per-dimension epochs and weighted losses
    /// per_dimension_epochs: [E/I, S/N, T/F, J/P]
    /// dimension_weights: [E/I, S/N, T/F, J/P] - higher weight = more importance
    pub fn train_per_dimension_weighted(
        &mut self,
        features: &[Vec<f64>],
        labels: &[String], // MBTI types like "INTJ", "ENFP"
        per_dimension_epochs: Vec<i64>,
        dimension_weights: Vec<f64>,
        batch_size: i64,
    ) {
        println!("🏋️  Training Multi-Task Model on {:?}...", self.device);
        println!("   4 binary classifiers with per-dimension epochs & weighted losses:");
        println!(
            "   - E/I: {} epochs, weight={:.1} (I: 77% vs E: 23%)",
            per_dimension_epochs[0], dimension_weights[0]
        );
        println!(
            "   - S/N: {} epochs, weight={:.1} (N: 86% vs S: 14%) <- Most imbalanced",
            per_dimension_epochs[1], dimension_weights[1]
        );
        println!(
            "   - T/F: {} epochs, weight={:.1} (F: 54% vs T: 46%) <- Weakest dimension",
            per_dimension_epochs[2], dimension_weights[2]
        );
        println!(
            "   - J/P: {} epochs, weight={:.1} (J: 60% vs P: 40%)\n",
            per_dimension_epochs[3], dimension_weights[3]
        );

        let n_samples = features.len() as i64;
        let feature_dim = features[0].len() as i64;

        // Convert MBTI labels to 4-dimension binary labels
        let (labels_ei, labels_sn, labels_tf, labels_jp) = self.parse_mbti_labels(labels);

        // Flatten features to 1D vec
        let flat_features: Vec<f32> = features
            .iter()
            .flat_map(|f| f.iter().map(|&x| x as f32))
            .collect();

        let x_train = Tensor::from_slice(&flat_features)
            .view([n_samples, feature_dim])
            .to(self.device);

        // Convert labels to tensors
        let y_ei = Tensor::from_slice(&labels_ei).to(self.device);
        let y_sn = Tensor::from_slice(&labels_sn).to(self.device);
        let y_tf = Tensor::from_slice(&labels_tf).to(self.device);
        let y_jp = Tensor::from_slice(&labels_jp).to(self.device);

        // Optimizer with weight decay (L2 regularization)
        let mut opt = nn::Adam {
            wd: self.weight_decay,
            ..Default::default()
        }
        .build(&self.vs, self.learning_rate)
        .unwrap();

        // Determine max epochs for the training loop
        let max_epochs = *per_dimension_epochs.iter().max().unwrap_or(&25);

        // Track which dimensions are still training
        let epochs_ei = per_dimension_epochs[0];
        let epochs_sn = per_dimension_epochs[1];
        let epochs_tf = per_dimension_epochs[2];
        let epochs_jp = per_dimension_epochs[3];

        // Training loop
        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            let mut correct_ei = 0;
            let mut correct_sn = 0;
            let mut correct_tf = 0;
            let mut correct_jp = 0;

            // Batch training
            for batch_start in (0..n_samples).step_by(batch_size as usize) {
                let batch_end = (batch_start + batch_size).min(n_samples);
                let batch_x = x_train.narrow(0, batch_start, batch_end - batch_start);
                let batch_y_ei = y_ei.narrow(0, batch_start, batch_end - batch_start);
                let batch_y_sn = y_sn.narrow(0, batch_start, batch_end - batch_start);
                let batch_y_tf = y_tf.narrow(0, batch_start, batch_end - batch_start);
                let batch_y_jp = y_jp.narrow(0, batch_start, batch_end - batch_start);

                // Forward pass through shared layers with dropout after each layer
                let mut shared_features = batch_x.shallow_clone();
                for layer in &self.shared_layers {
                    shared_features = layer.forward(&shared_features);
                    shared_features = shared_features.relu();
                    shared_features = shared_features.dropout(self.dropout_rate, true);
                }

                // Four independent predictions
                let logits_ei = self.head_ei.forward(&shared_features);
                let logits_sn = self.head_sn.forward(&shared_features);
                let logits_tf = self.head_tf.forward(&shared_features);
                let logits_jp = self.head_jp.forward(&shared_features);

                // Calculate weighted loss for each dimension (only for active dimensions)
                let mut weighted_loss = Tensor::zeros([], (tch::Kind::Float, self.device));
                let mut total_weight = 0.0;

                if epoch < epochs_ei {
                    let loss_ei = logits_ei.cross_entropy_for_logits(&batch_y_ei);
                    weighted_loss += &loss_ei * dimension_weights[0];
                    total_weight += dimension_weights[0];
                }
                if epoch < epochs_sn {
                    let loss_sn = logits_sn.cross_entropy_for_logits(&batch_y_sn);
                    weighted_loss += &loss_sn * dimension_weights[1];
                    total_weight += dimension_weights[1];
                }
                if epoch < epochs_tf {
                    let loss_tf = logits_tf.cross_entropy_for_logits(&batch_y_tf);
                    weighted_loss += &loss_tf * dimension_weights[2];
                    total_weight += dimension_weights[2];
                }
                if epoch < epochs_jp {
                    let loss_jp = logits_jp.cross_entropy_for_logits(&batch_y_jp);
                    weighted_loss += &loss_jp * dimension_weights[3];
                    total_weight += dimension_weights[3];
                }

                // Skip if no active dimensions
                if total_weight == 0.0 {
                    continue;
                }

                // Normalize by total weight
                let loss = weighted_loss / total_weight;

                opt.zero_grad();
                loss.backward();
                opt.step();

                // Calculate accuracy for this batch
                let pred_ei = logits_ei.argmax(-1, false);
                let pred_sn = logits_sn.argmax(-1, false);
                let pred_tf = logits_tf.argmax(-1, false);
                let pred_jp = logits_jp.argmax(-1, false);

                correct_ei += pred_ei
                    .eq_tensor(&batch_y_ei)
                    .sum(tch::Kind::Int64)
                    .int64_value(&[]);
                correct_sn += pred_sn
                    .eq_tensor(&batch_y_sn)
                    .sum(tch::Kind::Int64)
                    .int64_value(&[]);
                correct_tf += pred_tf
                    .eq_tensor(&batch_y_tf)
                    .sum(tch::Kind::Int64)
                    .int64_value(&[]);
                correct_jp += pred_jp
                    .eq_tensor(&batch_y_jp)
                    .sum(tch::Kind::Int64)
                    .int64_value(&[]);

                total_loss += loss.double_value(&[]);
            }

            if (epoch + 1) % 5 == 0 {
                let avg_loss = total_loss / (n_samples / batch_size) as f64;
                let acc_ei = correct_ei as f64 / n_samples as f64 * 100.0;
                let acc_sn = correct_sn as f64 / n_samples as f64 * 100.0;
                let acc_tf = correct_tf as f64 / n_samples as f64 * 100.0;
                let acc_jp = correct_jp as f64 / n_samples as f64 * 100.0;
                let avg_acc = (acc_ei + acc_sn + acc_tf + acc_jp) / 4.0;

                // Show which dimensions are still training
                let mut active_dims = Vec::new();
                if epoch < epochs_ei {
                    active_dims.push("E/I");
                }
                if epoch < epochs_sn {
                    active_dims.push("S/N");
                }
                if epoch < epochs_tf {
                    active_dims.push("T/F");
                }
                if epoch < epochs_jp {
                    active_dims.push("J/P");
                }
                let active_str = if active_dims.len() == 4 {
                    "all".to_string()
                } else if active_dims.is_empty() {
                    "done".to_string()
                } else {
                    active_dims.join("+")
                };

                println!(
                    "  Epoch {:3}/{}: Loss={:.4}, Avg={:.2}% (E/I={:.1}% S/N={:.1}% T/F={:.1}% J/P={:.1}%) [{}]",
                    epoch + 1,
                    max_epochs,
                    avg_loss,
                    avg_acc,
                    acc_ei,
                    acc_sn,
                    acc_tf,
                    acc_jp,
                    active_str
                );
            }
        }

        println!("✅ Multi-Task Training complete!\n");
    }

    /// Predict MBTI type from features
    pub fn predict(&self, features: &[f64]) -> String {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_slice(&features_f32)
            .to_device(self.device)
            .unsqueeze(0);

        // CRITICAL: Disable dropout during prediction by using no_grad
        tch::no_grad(|| {
            // Forward through shared layers WITHOUT dropout
            let mut shared_features = input_tensor.shallow_clone();
            for layer in &self.shared_layers {
                shared_features = layer.forward(&shared_features);
                shared_features = shared_features.relu();
                // No dropout during prediction!
            }

            // Get predictions from each head
            let logits_ei = self.head_ei.forward(&shared_features);
            let logits_sn = self.head_sn.forward(&shared_features);
            let logits_tf = self.head_tf.forward(&shared_features);
            let logits_jp = self.head_jp.forward(&shared_features);

            let pred_ei = logits_ei.argmax(-1, false).int64_value(&[0]);
            let pred_sn = logits_sn.argmax(-1, false).int64_value(&[0]);
            let pred_tf = logits_tf.argmax(-1, false).int64_value(&[0]);
            let pred_jp = logits_jp.argmax(-1, false).int64_value(&[0]);

            // Combine predictions into MBTI type
            let ei = if pred_ei == 0 { 'E' } else { 'I' };
            let sn = if pred_sn == 0 { 'S' } else { 'N' };
            let tf = if pred_tf == 0 { 'T' } else { 'F' };
            let jp = if pred_jp == 0 { 'J' } else { 'P' };

            format!("{}{}{}{}", ei, sn, tf, jp)
        })
    }

    /// Predict with confidence scores for each dimension
    /// Returns (prediction, [conf_ei, conf_sn, conf_tf, conf_jp])
    pub fn predict_with_confidence(&self, features: &[f64]) -> (String, [f64; 4]) {
        let features_f32: Vec<f32> = features.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_slice(&features_f32)
            .to_device(self.device)
            .unsqueeze(0);

        tch::no_grad(|| {
            // Forward through shared layers WITHOUT dropout
            let mut shared_features = input_tensor.shallow_clone();
            for layer in &self.shared_layers {
                shared_features = layer.forward(&shared_features);
                shared_features = shared_features.relu();
            }

            // Get logits from each head
            let logits_ei = self.head_ei.forward(&shared_features);
            let logits_sn = self.head_sn.forward(&shared_features);
            let logits_tf = self.head_tf.forward(&shared_features);
            let logits_jp = self.head_jp.forward(&shared_features);

            // Apply softmax to get probabilities
            let probs_ei = logits_ei.softmax(-1, tch::Kind::Float);
            let probs_sn = logits_sn.softmax(-1, tch::Kind::Float);
            let probs_tf = logits_tf.softmax(-1, tch::Kind::Float);
            let probs_jp = logits_jp.softmax(-1, tch::Kind::Float);

            // Get predictions and confidences (max probability)
            let pred_ei = logits_ei.argmax(-1, false).int64_value(&[0]);
            let pred_sn = logits_sn.argmax(-1, false).int64_value(&[0]);
            let pred_tf = logits_tf.argmax(-1, false).int64_value(&[0]);
            let pred_jp = logits_jp.argmax(-1, false).int64_value(&[0]);

            let conf_ei = probs_ei.max().double_value(&[]);
            let conf_sn = probs_sn.max().double_value(&[]);
            let conf_tf = probs_tf.max().double_value(&[]);
            let conf_jp = probs_jp.max().double_value(&[]);

            // Combine predictions into MBTI type
            let ei = if pred_ei == 0 { 'E' } else { 'I' };
            let sn = if pred_sn == 0 { 'S' } else { 'N' };
            let tf = if pred_tf == 0 { 'T' } else { 'F' };
            let jp = if pred_jp == 0 { 'J' } else { 'P' };

            (
                format!("{}{}{}{}", ei, sn, tf, jp),
                [conf_ei, conf_sn, conf_tf, conf_jp],
            )
        })
    }

    /// Batch prediction
    pub fn predict_batch(&self, features_batch: &[Vec<f64>]) -> Vec<String> {
        features_batch
            .iter()
            .map(|features| self.predict(features))
            .collect()
    }

    /// Batch prediction with confidence scores
    pub fn predict_batch_with_confidence(
        &self,
        features_batch: &[Vec<f64>],
    ) -> Vec<(String, [f64; 4])> {
        features_batch
            .iter()
            .map(|features| self.predict_with_confidence(features))
            .collect()
    }

    /// Parse MBTI labels into 4 binary dimensions
    fn parse_mbti_labels(&self, labels: &[String]) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
        let mut labels_ei = Vec::new();
        let mut labels_sn = Vec::new();
        let mut labels_tf = Vec::new();
        let mut labels_jp = Vec::new();

        for label in labels {
            let chars: Vec<char> = label.chars().collect();
            if chars.len() == 4 {
                // E=0, I=1
                labels_ei.push(if chars[0] == 'E' { 0 } else { 1 });
                // S=0, N=1
                labels_sn.push(if chars[1] == 'S' { 0 } else { 1 });
                // T=0, F=1
                labels_tf.push(if chars[2] == 'T' { 0 } else { 1 });
                // J=0, P=1
                labels_jp.push(if chars[3] == 'J' { 0 } else { 1 });
            }
        }

        (labels_ei, labels_sn, labels_tf, labels_jp)
    }

    /// Save model weights
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.vs.save(path)?;
        println!("  ✓ Multi-Task MLP weights saved to {}", path);
        Ok(())
    }

    /// Load model weights
    #[allow(dead_code)]
    pub fn load(
        path: &str,
        input_dim: i64,
        shared_layers: Vec<i64>,
        dropout_rate: f64,
    ) -> Result<Self, Box<dyn Error>> {
        let device = if tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let mut vs = nn::VarStore::new(device);
        vs.set_kind(tch::Kind::Float); // Use f32 to match input data type
        let root = vs.root();

        // Rebuild network structure (list of linear layers)
        let mut shared_linear_layers = Vec::new();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in shared_layers.iter().enumerate() {
            let linear = nn::linear(
                &root / format!("shared_fc{}", i),
                current_dim,
                hidden_dim,
                Default::default(),
            );
            shared_linear_layers.push(linear);
            current_dim = hidden_dim;
        }

        let head_ei = nn::linear(&root / "head_ei", current_dim, 2, Default::default());
        let head_sn = nn::linear(&root / "head_sn", current_dim, 2, Default::default());
        let head_tf = nn::linear(&root / "head_tf", current_dim, 2, Default::default());
        let head_jp = nn::linear(&root / "head_jp", current_dim, 2, Default::default());

        // Load weights
        vs.load(path)?;

        Ok(MultiTaskGpuMLP {
            device,
            vs,
            shared_layers: shared_linear_layers,
            head_ei,
            head_sn,
            head_tf,
            head_jp,
            learning_rate: 0.001,
            weight_decay: 0.0, // Default for loaded models
            dropout_rate,
        })
    }
}
