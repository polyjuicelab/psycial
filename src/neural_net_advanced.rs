/// Advanced MLP with Adam optimizer and Dropout
/// Production-grade neural network implementation

use rand::Rng;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct AdvancedMLP {
    output_dim: usize,
    
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    
    // Adam optimizer state
    m_weights: Vec<Vec<Vec<f64>>>,
    v_weights: Vec<Vec<Vec<f64>>>,
    m_biases: Vec<Vec<f64>>,
    v_biases: Vec<Vec<f64>>,
    
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
    
    dropout_rate: f64,
    
    class_to_idx: HashMap<String, usize>,
    idx_to_class: HashMap<usize, String>,
}

impl AdvancedMLP {
    pub fn num_layers(&self) -> usize {
        self.weights.len()
    }
    
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize, learning_rate: f64, dropout_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut m_weights = Vec::new();
        let mut v_weights = Vec::new();
        let mut m_biases = Vec::new();
        let mut v_biases = Vec::new();
        
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&hidden_dims);
        layer_sizes.push(output_dim);
        
        // He initialization (better for ReLU)
        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            let std = (2.0 / n_in as f64).sqrt();
            
            let mut layer_weights = Vec::new();
            let mut layer_m_w = Vec::new();
            let mut layer_v_w = Vec::new();
            
            for _ in 0..n_out {
                let neuron_weights: Vec<f64> = (0..n_in)
                    .map(|_| rng.gen_range(-std..std))
                    .collect();
                layer_weights.push(neuron_weights.clone());
                layer_m_w.push(vec![0.0; n_in]);
                layer_v_w.push(vec![0.0; n_in]);
            }
            weights.push(layer_weights);
            m_weights.push(layer_m_w);
            v_weights.push(layer_v_w);
            
            let layer_biases: Vec<f64> = vec![0.0; n_out];
            biases.push(layer_biases.clone());
            m_biases.push(vec![0.0; n_out]);
            v_biases.push(vec![0.0; n_out]);
        }
        
        AdvancedMLP {
            output_dim,
            weights,
            biases,
            m_weights,
            v_weights,
            m_biases,
            v_biases,
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            timestep: 0,
            dropout_rate,
            class_to_idx: HashMap::new(),
            idx_to_class: HashMap::new(),
        }
    }
    
    pub fn train(&mut self, features: &[Vec<f64>], labels: &[String], epochs: usize, batch_size: usize) {
        // Build class mapping
        let mut unique_classes: Vec<String> = labels.iter().cloned().collect();
        unique_classes.sort();
        unique_classes.dedup();
        
        for (idx, class) in unique_classes.iter().enumerate() {
            self.class_to_idx.insert(class.clone(), idx);
            self.idx_to_class.insert(idx, class.clone());
        }
        
        println!("Training Advanced MLP:");
        println!("  Optimizer: Adam (beta1={}, beta2={})", self.beta1, self.beta2);
        println!("  Dropout: {}", self.dropout_rate);
        println!("  Epochs: {} | Batch size: {}\n", epochs, batch_size);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;
            
            for batch_start in (0..features.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(features.len());
                
                for i in batch_start..batch_end {
                    self.timestep += 1;
                    
                    let target_idx = self.class_to_idx[&labels[i]];
                    let mut target = vec![0.0; self.output_dim];
                    target[target_idx] = 1.0;
                    
                    // Forward with dropout
                    let (activations, _) = self.forward_train(&features[i]);
                    let output = activations.last().unwrap();
                    
                    // Loss
                    let loss: f64 = output.iter()
                        .zip(target.iter())
                        .map(|(&o, &t)| -t * o.max(1e-10).ln())
                        .sum();
                    total_loss += loss;
                    
                    // Accuracy
                    let pred_idx = output.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    if pred_idx == target_idx {
                        correct += 1;
                    }
                    
                    // Backward with Adam
                    self.backward_adam(&features[i], &target);
                }
            }
            
            if (epoch + 1) % 5 == 0 || epoch == 0 {
                let avg_loss = total_loss / features.len() as f64;
                let accuracy = correct as f64 / features.len() as f64;
                println!("  Epoch {:>3}/{}: Loss={:.4}, Acc={:.2}%", 
                         epoch + 1, epochs, avg_loss, accuracy * 100.0);
            }
        }
        
        println!("\nTraining complete!");
    }
    
    fn forward_train(&self, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut rng = rand::thread_rng();
        let mut activations = vec![input.to_vec()];
        let mut pre_activations = Vec::new();
        
        for layer_idx in 0..self.weights.len() {
            let prev_activation = activations.last().unwrap();
            let mut pre_activation = Vec::new();
            let mut activation = Vec::new();
            
            for neuron_idx in 0..self.weights[layer_idx].len() {
                let weights = &self.weights[layer_idx][neuron_idx];
                let bias = self.biases[layer_idx][neuron_idx];
                
                let z: f64 = weights.iter()
                    .zip(prev_activation.iter())
                    .map(|(&w, &a)| w * a)
                    .sum::<f64>() + bias;
                
                pre_activation.push(z);
                
                let mut a = if layer_idx < self.weights.len() - 1 {
                    z.max(0.0) // ReLU
                } else {
                    z // Will apply softmax
                };
                
                // Dropout (only hidden layers, only during training)
                if layer_idx < self.weights.len() - 1 {
                    if rng.gen::<f64>() < self.dropout_rate {
                        a = 0.0;
                    } else {
                        a /= 1.0 - self.dropout_rate; // Inverted dropout
                    }
                }
                
                activation.push(a);
            }
            
            if layer_idx == self.weights.len() - 1 {
                activation = self.softmax(&activation);
            }
            
            pre_activations.push(pre_activation);
            activations.push(activation);
        }
        
        (activations, pre_activations)
    }
    
    fn forward_test(&self, input: &[f64]) -> Vec<Vec<f64>> {
        let mut activations = vec![input.to_vec()];
        
        for layer_idx in 0..self.weights.len() {
            let prev_activation = activations.last().unwrap();
            let mut activation = Vec::new();
            
            for neuron_idx in 0..self.weights[layer_idx].len() {
                let weights = &self.weights[layer_idx][neuron_idx];
                let bias = self.biases[layer_idx][neuron_idx];
                
                let z: f64 = weights.iter()
                    .zip(prev_activation.iter())
                    .map(|(&w, &a)| w * a)
                    .sum::<f64>() + bias;
                
                let a = if layer_idx < self.weights.len() - 1 {
                    z.max(0.0)
                } else {
                    z
                };
                
                activation.push(a);
            }
            
            if layer_idx == self.weights.len() - 1 {
                activation = self.softmax(&activation);
            }
            
            activations.push(activation);
        }
        
        activations
    }
    
    fn backward_adam(&mut self, input: &[f64], target: &[f64]) {
        let (activations, pre_activations) = self.forward_train(input);
        
        let output = activations.last().unwrap();
        let mut delta: Vec<f64> = output.iter()
            .zip(target.iter())
            .map(|(&o, &t)| o - t)
            .collect();
        
        for layer_idx in (0..self.weights.len()).rev() {
            let prev_activation = &activations[layer_idx];
            
            // Update weights with Adam
            for neuron_idx in 0..self.weights[layer_idx].len() {
                for weight_idx in 0..self.weights[layer_idx][neuron_idx].len() {
                    let gradient = delta[neuron_idx] * prev_activation[weight_idx];
                    
                    // Adam update for weight
                    self.m_weights[layer_idx][neuron_idx][weight_idx] = 
                        self.beta1 * self.m_weights[layer_idx][neuron_idx][weight_idx] + (1.0 - self.beta1) * gradient;
                    self.v_weights[layer_idx][neuron_idx][weight_idx] = 
                        self.beta2 * self.v_weights[layer_idx][neuron_idx][weight_idx] + (1.0 - self.beta2) * gradient * gradient;
                    
                    let m_hat = self.m_weights[layer_idx][neuron_idx][weight_idx] / (1.0 - self.beta1.powi(self.timestep as i32));
                    let v_hat = self.v_weights[layer_idx][neuron_idx][weight_idx] / (1.0 - self.beta2.powi(self.timestep as i32));
                    
                    self.weights[layer_idx][neuron_idx][weight_idx] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
                }
                
                // Adam update for bias
                let gradient = delta[neuron_idx];
                self.m_biases[layer_idx][neuron_idx] = 
                    self.beta1 * self.m_biases[layer_idx][neuron_idx] + (1.0 - self.beta1) * gradient;
                self.v_biases[layer_idx][neuron_idx] = 
                    self.beta2 * self.v_biases[layer_idx][neuron_idx] + (1.0 - self.beta2) * gradient * gradient;
                
                let m_hat = self.m_biases[layer_idx][neuron_idx] / (1.0 - self.beta1.powi(self.timestep as i32));
                let v_hat = self.v_biases[layer_idx][neuron_idx] / (1.0 - self.beta2.powi(self.timestep as i32));
                
                self.biases[layer_idx][neuron_idx] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
            }
            
            // Backpropagate error
            if layer_idx > 0 {
                let mut new_delta = vec![0.0; prev_activation.len()];
                for prev_idx in 0..prev_activation.len() {
                    let mut error = 0.0;
                    for neuron_idx in 0..delta.len() {
                        error += self.weights[layer_idx][neuron_idx][prev_idx] * delta[neuron_idx];
                    }
                    let z = pre_activations[layer_idx - 1][prev_idx];
                    new_delta[prev_idx] = if z > 0.0 { error } else { 0.0 };
                }
                delta = new_delta;
            }
        }
    }
    
    pub fn predict(&self, features: &[f64]) -> String {
        let activations = self.forward_test(features);
        let output = activations.last().unwrap();
        
        let pred_idx = output.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        self.idx_to_class.get(&pred_idx)
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".to_string())
    }
    
    fn softmax(&self, x: &[f64]) -> Vec<f64> {
        let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        exp.iter().map(|&v| v / sum).collect()
    }
    
    /// Save model to JSON file
    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)?;
        Ok(())
    }
    
    /// Load model from JSON file
    #[allow(dead_code)]
    pub fn load(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let model = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(model)
    }
}

