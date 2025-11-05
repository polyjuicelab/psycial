/// Multi-task GPU-accelerated MLP for MBTI 4-dimension prediction
/// Each dimension (E/I, S/N, T/F, J/P) is predicted independently
use std::error::Error;

#[cfg(feature = "bert")]
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

#[cfg(feature = "bert")]
pub struct MultiTaskGpuMLP {
    device: Device,
    vs: nn::VarStore,

    // Shared feature extraction layers
    shared_net: nn::Sequential,

    // Four independent output heads for each dimension
    head_ei: nn::Linear, // Extraversion/Introversion
    head_sn: nn::Linear, // Sensing/Intuition
    head_tf: nn::Linear, // Thinking/Feeling
    head_jp: nn::Linear, // Judging/Perceiving

    learning_rate: f64,
    weight_decay: f64,
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

        // Build shared network layers
        let mut shared_net = nn::seq();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in shared_layers.iter().enumerate() {
            shared_net = shared_net
                .add(nn::linear(
                    &root / format!("shared_fc{}", i),
                    current_dim,
                    hidden_dim,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add_fn(move |x| x.dropout(dropout_rate, true));
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
            shared_net,
            head_ei,
            head_sn,
            head_tf,
            head_jp,
            learning_rate,
            weight_decay,
        }
    }

    /// Train the multi-task model
    pub fn train(
        &mut self,
        features: &[Vec<f64>],
        labels: &[String], // MBTI types like "INTJ", "ENFP"
        epochs: i64,
        batch_size: i64,
    ) {
        println!("🏋️  Training Multi-Task Model on {:?}...", self.device);
        println!("   4 binary classifiers: E/I, S/N, T/F, J/P\n");

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

        // Training loop
        for epoch in 0..epochs {
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

                // Forward pass through shared layers
                let shared_features = self.shared_net.forward(&batch_x);

                // Four independent predictions
                let logits_ei = self.head_ei.forward(&shared_features);
                let logits_sn = self.head_sn.forward(&shared_features);
                let logits_tf = self.head_tf.forward(&shared_features);
                let logits_jp = self.head_jp.forward(&shared_features);

                // Calculate loss for each dimension
                let loss_ei = logits_ei.cross_entropy_for_logits(&batch_y_ei);
                let loss_sn = logits_sn.cross_entropy_for_logits(&batch_y_sn);
                let loss_tf = logits_tf.cross_entropy_for_logits(&batch_y_tf);
                let loss_jp = logits_jp.cross_entropy_for_logits(&batch_y_jp);

                // Total loss (average of 4 dimensions)
                let loss = (&loss_ei + &loss_sn + &loss_tf + &loss_jp) / 4.0;

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

                println!(
                    "  Epoch {:3}/{}: Loss={:.4}, Avg={:.2}% (E/I={:.1}% S/N={:.1}% T/F={:.1}% J/P={:.1}%)",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    avg_acc,
                    acc_ei,
                    acc_sn,
                    acc_tf,
                    acc_jp
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

        // Forward through shared layers (in eval mode)
        let shared_features = tch::no_grad(|| self.shared_net.forward(&input_tensor));

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
    }

    /// Batch prediction
    pub fn predict_batch(&self, features_batch: &[Vec<f64>]) -> Vec<String> {
        features_batch
            .iter()
            .map(|features| self.predict(features))
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

        // Rebuild network structure
        let mut shared_net = nn::seq();
        let mut current_dim = input_dim;

        for (i, &hidden_dim) in shared_layers.iter().enumerate() {
            shared_net = shared_net
                .add(nn::linear(
                    &root / format!("shared_fc{}", i),
                    current_dim,
                    hidden_dim,
                    Default::default(),
                ))
                .add_fn(|x| x.relu())
                .add_fn(move |x| x.dropout(dropout_rate, false));
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
            shared_net,
            head_ei,
            head_sn,
            head_tf,
            head_jp,
            learning_rate: 0.001,
            weight_decay: 0.0, // Default for loaded models
        })
    }
}
