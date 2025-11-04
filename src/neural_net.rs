/// Simple MLP neural network classifier
/// Pure Rust implementation with backpropagation

use rand::Rng;
use std::collections::HashMap;

pub struct MLPClassifier {
    // Network architecture
    #[allow(dead_code)]
    input_dim: usize,
    #[allow(dead_code)]
    hidden_dims: Vec<usize>,
    output_dim: usize,
    
    // Weights and biases
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    
    // Training parameters
    learning_rate: f64,
    
    // Class mapping
    class_to_idx: HashMap<String, usize>,
    idx_to_class: HashMap<usize, String>,
}

impl MLPClassifier {
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        let mut layer_sizes = vec![input_dim];
        layer_sizes.extend(&hidden_dims);
        layer_sizes.push(output_dim);
        
        // Xavier initialization
        for i in 0..layer_sizes.len() - 1 {
            let n_in = layer_sizes[i];
            let n_out = layer_sizes[i + 1];
            let limit = (6.0 / (n_in + n_out) as f64).sqrt();
            
            let mut layer_weights = Vec::new();
            for _ in 0..n_out {
                let neuron_weights: Vec<f64> = (0..n_in)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect();
                layer_weights.push(neuron_weights);
            }
            weights.push(layer_weights);
            
            let layer_biases: Vec<f64> = (0..n_out)
                .map(|_| rng.gen_range(-limit..limit))
                .collect();
            biases.push(layer_biases);
        }
        
        MLPClassifier {
            input_dim,
            hidden_dims,
            output_dim,
            weights,
            biases,
            learning_rate,
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
        
        println!("Training MLP: {} epochs, batch size {}", epochs, batch_size);
        
        for epoch in 0..epochs {
            let mut total_loss = 0.0;
            let mut correct = 0;
            
            // Mini-batch training
            for batch_start in (0..features.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(features.len());
                
                for i in batch_start..batch_end {
                    let target_idx = self.class_to_idx[&labels[i]];
                    let mut target = vec![0.0; self.output_dim];
                    target[target_idx] = 1.0;
                    
                    // Forward pass
                    let (activations, _) = self.forward(&features[i]);
                    let output = activations.last().unwrap();
                    
                    // Calculate loss
                    let loss: f64 = output.iter()
                        .zip(target.iter())
                        .map(|(&o, &t)| -t * o.ln().max(-100.0))
                        .sum();
                    total_loss += loss;
                    
                    // Check accuracy
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
                    
                    // Backward pass
                    self.backward(&features[i], &target);
                }
            }
            
            if (epoch + 1) % 5 == 0 {
                let avg_loss = total_loss / features.len() as f64;
                let accuracy = correct as f64 / features.len() as f64;
                println!("  Epoch {}/{}: Loss={:.4}, Acc={:.2}%", 
                         epoch + 1, epochs, avg_loss, accuracy * 100.0);
            }
        }
        
        println!("Training complete!");
    }
    
    fn forward(&self, input: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
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
                
                // Activation: ReLU for hidden, Softmax for output
                let a = if layer_idx < self.weights.len() - 1 {
                    z.max(0.0) // ReLU
                } else {
                    z // Will apply softmax to final layer
                };
                activation.push(a);
            }
            
            // Softmax for final layer
            if layer_idx == self.weights.len() - 1 {
                activation = self.softmax(&activation);
            }
            
            pre_activations.push(pre_activation);
            activations.push(activation);
        }
        
        (activations, pre_activations)
    }
    
    fn backward(&mut self, input: &[f64], target: &[f64]) {
        let (activations, pre_activations) = self.forward(input);
        
        // Calculate output layer error
        let output = activations.last().unwrap();
        let mut delta: Vec<f64> = output.iter()
            .zip(target.iter())
            .map(|(&o, &t)| o - t)
            .collect();
        
        // Backpropagate
        for layer_idx in (0..self.weights.len()).rev() {
            let prev_activation = &activations[layer_idx];
            
            // Update weights and biases
            for neuron_idx in 0..self.weights[layer_idx].len() {
                for weight_idx in 0..self.weights[layer_idx][neuron_idx].len() {
                    let gradient = delta[neuron_idx] * prev_activation[weight_idx];
                    self.weights[layer_idx][neuron_idx][weight_idx] -= self.learning_rate * gradient;
                }
                self.biases[layer_idx][neuron_idx] -= self.learning_rate * delta[neuron_idx];
            }
            
            // Calculate error for previous layer
            if layer_idx > 0 {
                let mut new_delta = vec![0.0; prev_activation.len()];
                for prev_idx in 0..prev_activation.len() {
                    let mut error = 0.0;
                    for neuron_idx in 0..delta.len() {
                        error += self.weights[layer_idx][neuron_idx][prev_idx] * delta[neuron_idx];
                    }
                    // ReLU derivative
                    let z = pre_activations[layer_idx - 1][prev_idx];
                    new_delta[prev_idx] = if z > 0.0 { error } else { 0.0 };
                }
                delta = new_delta;
            }
        }
    }
    
    pub fn predict(&self, features: &[f64]) -> String {
        let (activations, _) = self.forward(features);
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mlp_creation() {
        let mlp = MLPClassifier::new(100, vec![64, 32], 16, 0.01);
        assert_eq!(mlp.input_dim, 100);
        assert_eq!(mlp.output_dim, 16);
    }
    
    #[test]
    fn test_forward_pass() {
        let mlp = MLPClassifier::new(10, vec![5], 3, 0.01);
        let input = vec![0.5; 10];
        let (activations, _) = mlp.forward(&input);
        
        assert_eq!(activations.len(), 3); // input + hidden + output
        assert_eq!(activations.last().unwrap().len(), 3);
        
        // Softmax should sum to 1
        let sum: f64 = activations.last().unwrap().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

