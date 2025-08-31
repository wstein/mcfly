use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Scalable 3-layer MLP with built-in feature normalization and bold highlighting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalableMlp {
    // Network architecture (3-layer: input -> hidden1 -> hidden2 -> output)
    pub input_dim: usize,
    pub hidden1_dim: usize,
    pub hidden2_dim: usize,
    pub output_dim: usize,
    
    // Layer weights and biases
    pub w1: Vec<f64>, // input -> hidden1
    pub b1: Vec<f64>,
    pub w2: Vec<f64>, // hidden1 -> hidden2  
    pub b2: Vec<f64>,
    pub w3: Vec<f64>, // hidden2 -> output
    pub b3: Vec<f64>,
    
    // Built-in feature normalization
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub feature_count: usize,
    
    // Adam optimizer state
    pub update_count: usize,
    #[serde(skip)]
    pub adam_m_w1: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_w1: Option<Vec<f64>>,
    #[serde(skip)]
    pub adam_m_b1: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_b1: Option<Vec<f64>>,
    #[serde(skip)]
    pub adam_m_w2: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_w2: Option<Vec<f64>>,
    #[serde(skip)]
    pub adam_m_b2: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_b2: Option<Vec<f64>>,
    #[serde(skip)]
    pub adam_m_w3: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_w3: Option<Vec<f64>>,
    #[serde(skip)]
    pub adam_m_b3: Option<Vec<f64>>, 
    #[serde(skip)]
    pub adam_v_b3: Option<Vec<f64>>,
}

impl ScalableMlp {
    /// Create a new 3-layer MLP with Xavier initialization for better gradient flow
    pub fn new(input_dim: usize, hidden1_dim: usize, hidden2_dim: usize, output_dim: usize) -> Self {
        // Xavier initialization: weights ~ N(0, sqrt(2/(fan_in + fan_out)))
        let init_w1 = (2.0 / (input_dim + hidden1_dim) as f64).sqrt() * 0.1;
        let init_w2 = (2.0 / (hidden1_dim + hidden2_dim) as f64).sqrt() * 0.1;
        let init_w3 = (2.0 / (hidden2_dim + output_dim) as f64).sqrt() * 0.1;
        
        let w1 = vec![init_w1; input_dim * hidden1_dim];
        let b1 = vec![0.0; hidden1_dim];
        let w2 = vec![init_w2; hidden1_dim * hidden2_dim];
        let b2 = vec![0.0; hidden2_dim];
        let w3 = vec![init_w3; hidden2_dim * output_dim];
        let b3 = vec![0.0; output_dim];
        
        Self {
            input_dim,
            hidden1_dim,
            hidden2_dim,
            output_dim,
            w1, b1, w2, b2, w3, b3,
            feature_means: vec![0.0; input_dim],
            feature_stds: vec![1.0; input_dim],
            feature_count: 0,
            update_count: 0,
            adam_m_w1: None, adam_v_w1: None, adam_m_b1: None, adam_v_b1: None,
            adam_m_w2: None, adam_v_w2: None, adam_m_b2: None, adam_v_b2: None,
            adam_m_w3: None, adam_v_w3: None, adam_m_b3: None, adam_v_b3: None,
        }
    }
    
    /// Update running feature statistics (for built-in normalization)
    fn update_feature_stats(&mut self, features: &[f64]) {
        assert_eq!(features.len(), self.input_dim);
        
        self.feature_count += 1;
        let n = self.feature_count as f64;
        
        // Online update of mean and variance (Welford's algorithm)
        for i in 0..self.input_dim {
            let old_mean = self.feature_means[i];
            self.feature_means[i] += (features[i] - old_mean) / n;
            
            if self.feature_count > 1 {
                let old_var = if self.feature_stds[i] == 1.0 { 0.0 } else { self.feature_stds[i].powi(2) };
                let new_var = ((n - 1.0) * old_var + (features[i] - old_mean) * (features[i] - self.feature_means[i])) / n;
                self.feature_stds[i] = (new_var + 1e-8).sqrt(); // Add small epsilon for numerical stability
            }
        }
    }
    
    /// Normalize features using running statistics
    fn normalize_features(&self, features: &[f64]) -> Vec<f64> {
        features.iter()
            .zip(&self.feature_means)
            .zip(&self.feature_stds)
            .map(|((f, m), s)| (f - m) / s)
            .collect()
    }
    
    /// Forward pass returning raw scores (no manual scaling)
    pub fn score(&self, features: &[f64]) -> Vec<f64> {
        assert_eq!(features.len(), self.input_dim);
        
        // Normalize features if we have sufficient statistics
        let normalized = if self.feature_count > 10 {
            self.normalize_features(features)
        } else {
            features.to_vec()
        };
        
        // Layer 1: input -> hidden1 (ReLU activation)
        let mut h1 = vec![0.0; self.hidden1_dim];
        for i in 0..self.hidden1_dim {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += normalized[j] * self.w1[j * self.hidden1_dim + i];
            }
            h1[i] = sum.max(0.0); // ReLU
        }
        
        // Layer 2: hidden1 -> hidden2 (ReLU activation)
        let mut h2 = vec![0.0; self.hidden2_dim];
        for i in 0..self.hidden2_dim {
            let mut sum = self.b2[i];
            for j in 0..self.hidden1_dim {
                sum += h1[j] * self.w2[j * self.hidden2_dim + i];
            }
            h2[i] = sum.max(0.0); // ReLU
        }
        
        // Layer 3: hidden2 -> output (linear activation for raw scores)
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = self.b3[i];
            for j in 0..self.hidden2_dim {
                sum += h2[j] * self.w3[j * self.output_dim + i];
            }
            output[i] = sum; // Raw output score
        }
        
        output
    }
    
    /// Initialize Adam optimizer state
    pub fn optimizer_init(&mut self) {
        self.adam_m_w1 = Some(vec![0.0; self.w1.len()]);
        self.adam_v_w1 = Some(vec![0.0; self.w1.len()]);
        self.adam_m_b1 = Some(vec![0.0; self.b1.len()]);
        self.adam_v_b1 = Some(vec![0.0; self.b1.len()]);
        self.adam_m_w2 = Some(vec![0.0; self.w2.len()]);
        self.adam_v_w2 = Some(vec![0.0; self.w2.len()]);
        self.adam_m_b2 = Some(vec![0.0; self.b2.len()]);
        self.adam_v_b2 = Some(vec![0.0; self.b2.len()]);
        self.adam_m_w3 = Some(vec![0.0; self.w3.len()]);
        self.adam_v_w3 = Some(vec![0.0; self.w3.len()]);
        self.adam_m_b3 = Some(vec![0.0; self.b3.len()]);
        self.adam_v_b3 = Some(vec![0.0; self.b3.len()]);
    }
    
    /// Single SGD update with Adam optimizer and gradient clipping
    pub fn update(&mut self, features: &[f64], target: &[f64], lr: f64, weight_decay: f64) {
        assert_eq!(features.len(), self.input_dim);
        assert_eq!(target.len(), self.output_dim);
        
        // Update feature statistics
        self.update_feature_stats(features);
        
        // Normalize features
        let normalized = if self.feature_count > 10 {
            self.normalize_features(features)
        } else {
            features.to_vec()
        };
        
        // Forward pass with intermediate values saved for backprop
        let mut h1 = vec![0.0; self.hidden1_dim];
        for i in 0..self.hidden1_dim {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += normalized[j] * self.w1[j * self.hidden1_dim + i];
            }
            h1[i] = sum.max(0.0); // ReLU
        }
        
        let mut h2 = vec![0.0; self.hidden2_dim];
        for i in 0..self.hidden2_dim {
            let mut sum = self.b2[i];
            for j in 0..self.hidden1_dim {
                sum += h1[j] * self.w2[j * self.hidden2_dim + i];
            }
            h2[i] = sum.max(0.0); // ReLU
        }
        
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = self.b3[i];
            for j in 0..self.hidden2_dim {
                sum += h2[j] * self.w3[j * self.output_dim + i];
            }
            output[i] = sum;
        }
        
        // Compute gradients with clipping for stability
        let max_grad = 1.0;
        
        // Output gradient
        let mut grad_output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let g = 2.0 * (output[i] - target[i]) / self.output_dim as f64;
            grad_output[i] = g.clamp(-max_grad, max_grad);
        }
        
        // Initialize Adam if needed
        if self.adam_m_w1.is_none() {
            self.optimizer_init();
        }
        
        self.update_count += 1;
        
        // Backprop and update output layer (w3/b3)
        self.update_output_layer(&h2, &grad_output, lr, weight_decay);
        
        // Backprop to hidden2
        let mut grad_h2 = vec![0.0; self.hidden2_dim];
        for j in 0..self.hidden2_dim {
            for i in 0..self.output_dim {
                grad_h2[j] += grad_output[i] * self.w3[j * self.output_dim + i];
            }
            grad_h2[j] = if h2[j] > 0.0 { grad_h2[j] } else { 0.0 }; // ReLU derivative
        }
        
        // Update hidden2 layer (w2/b2)
        self.update_hidden2_layer(&h1, &grad_h2, lr, weight_decay);
        
        // Backprop to hidden1
        let mut grad_h1 = vec![0.0; self.hidden1_dim];
        for j in 0..self.hidden1_dim {
            for i in 0..self.hidden2_dim {
                grad_h1[j] += grad_h2[i] * self.w2[j * self.hidden2_dim + i];
            }
            grad_h1[j] = if h1[j] > 0.0 { grad_h1[j] } else { 0.0 }; // ReLU derivative
        }
        
        // Update hidden1 layer (w1/b1)
        self.update_hidden1_layer(&normalized, &grad_h1, lr, weight_decay);
    }
    
    /// Batch update for efficiency during training
    pub fn update_batch(&mut self, batch_inputs: &[Vec<f64>], batch_targets: &[Vec<f64>], lr: f64, weight_decay: f64) {
        for (inputs, targets) in batch_inputs.iter().zip(batch_targets.iter()) {
            self.update(inputs, targets, lr, weight_decay);
        }
    }
    
    /// Update output layer (w3/b3)
    fn update_output_layer(&mut self, h2: &[f64], grad_output: &[f64], lr: f64, weight_decay: f64) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let t = self.update_count as f64;
        
        let m_w = self.adam_m_w3.as_mut().unwrap();
        let v_w = self.adam_v_w3.as_mut().unwrap();
        let m_b = self.adam_m_b3.as_mut().unwrap();
        let v_b = self.adam_v_b3.as_mut().unwrap();
        
        // Update biases
        for i in 0..self.b3.len() {
            let g = grad_output[i] + weight_decay * self.b3[i];
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g;
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = m_b[i] / (1.0 - beta1.powf(t));
            let v_hat = v_b[i] / (1.0 - beta2.powf(t));
            self.b3[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        
        // Update weights
        for j in 0..h2.len() {
            for i in 0..self.output_dim {
                let idx = j * self.output_dim + i;
                let g = grad_output[i] * h2[j] + weight_decay * self.w3[idx];
                m_w[idx] = beta1 * m_w[idx] + (1.0 - beta1) * g;
                v_w[idx] = beta2 * v_w[idx] + (1.0 - beta2) * g * g;
                let m_hat = m_w[idx] / (1.0 - beta1.powf(t));
                let v_hat = v_w[idx] / (1.0 - beta2.powf(t));
                self.w3[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
    
    /// Update hidden2 layer (w2/b2)
    fn update_hidden2_layer(&mut self, h1: &[f64], grad_h2: &[f64], lr: f64, weight_decay: f64) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let t = self.update_count as f64;
        
        let m_w = self.adam_m_w2.as_mut().unwrap();
        let v_w = self.adam_v_w2.as_mut().unwrap();
        let m_b = self.adam_m_b2.as_mut().unwrap();
        let v_b = self.adam_v_b2.as_mut().unwrap();
        
        // Update biases
        for i in 0..self.b2.len() {
            let g = grad_h2[i] + weight_decay * self.b2[i];
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g;
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = m_b[i] / (1.0 - beta1.powf(t));
            let v_hat = v_b[i] / (1.0 - beta2.powf(t));
            self.b2[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        
        // Update weights
        for j in 0..h1.len() {
            for i in 0..self.hidden2_dim {
                let idx = j * self.hidden2_dim + i;
                let g = grad_h2[i] * h1[j] + weight_decay * self.w2[idx];
                m_w[idx] = beta1 * m_w[idx] + (1.0 - beta1) * g;
                v_w[idx] = beta2 * v_w[idx] + (1.0 - beta2) * g * g;
                let m_hat = m_w[idx] / (1.0 - beta1.powf(t));
                let v_hat = v_w[idx] / (1.0 - beta2.powf(t));
                self.w2[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
    
    /// Update hidden1 layer (w1/b1)
    fn update_hidden1_layer(&mut self, input: &[f64], grad_h1: &[f64], lr: f64, weight_decay: f64) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;
        let t = self.update_count as f64;
        
        let m_w = self.adam_m_w1.as_mut().unwrap();
        let v_w = self.adam_v_w1.as_mut().unwrap();
        let m_b = self.adam_m_b1.as_mut().unwrap();
        let v_b = self.adam_v_b1.as_mut().unwrap();
        
        // Update biases
        for i in 0..self.b1.len() {
            let g = grad_h1[i] + weight_decay * self.b1[i];
            m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * g;
            v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = m_b[i] / (1.0 - beta1.powf(t));
            let v_hat = v_b[i] / (1.0 - beta2.powf(t));
            self.b1[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        
        // Update weights
        for j in 0..input.len() {
            for i in 0..self.hidden1_dim {
                let idx = j * self.hidden1_dim + i;
                let g = grad_h1[i] * input[j] + weight_decay * self.w1[idx];
                m_w[idx] = beta1 * m_w[idx] + (1.0 - beta1) * g;
                v_w[idx] = beta2 * v_w[idx] + (1.0 - beta2) * g * g;
                let m_hat = m_w[idx] / (1.0 - beta1.powf(t));
                let v_hat = v_w[idx] / (1.0 - beta2.powf(t));
                self.w1[idx] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }
    
    /// Save model to YAML file
    pub fn save(&self, path: &PathBuf) -> Result<(), std::io::Error> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let s = serde_yaml::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, s)
    }
    
    /// Load model from YAML file
    pub fn load(path: &PathBuf) -> Option<Self> {
        match fs::read_to_string(path) {
            Ok(s) => serde_yaml::from_str(&s).ok(),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_3_layer_network() {
        let mut model = ScalableMlp::new(10, 16, 8, 1);
        let features = vec![0.1; 10];
        let before = model.score(&features);
        model.update(&features, &[1.0], 0.01, 1e-4);
        let after = model.score(&features);
        assert_ne!(before, after);
        assert!(model.w1.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_feature_normalization() {
        let mut model = ScalableMlp::new(2, 4, 2, 1);
        
        // Add several samples to build statistics
        for i in 0..20 {
            let features = vec![i as f64, (i * 2) as f64];
            model.update(&features, &[1.0], 0.01, 1e-4);
        }
        
        // Feature stats should be updated
        assert!(model.feature_count == 20);
        assert!(model.feature_means[0] > 0.0);
        assert!(model.feature_stds[0] > 0.0);
    }

    #[test]
    fn test_save_and_load() {
        let model = ScalableMlp::new(5, 8, 4, 1);
        let tmp_dir = tempdir().unwrap();
        let path = tmp_dir.path().join("test_model.yaml");
        
        model.save(&path).unwrap();
        let loaded = ScalableMlp::load(&path).unwrap();
        
        assert_eq!(model.input_dim, loaded.input_dim);
        assert_eq!(model.hidden1_dim, loaded.hidden1_dim);
        assert_eq!(model.hidden2_dim, loaded.hidden2_dim);
        assert_eq!(model.output_dim, loaded.output_dim);
    }
}
