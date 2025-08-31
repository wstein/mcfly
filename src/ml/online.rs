use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;

// Keep the model very small: single hidden layer MLP
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SimpleMlp {
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    /// Counts how many online updates have been applied. Persisted with the model YAML.
    pub update_count: u32,
    // Optional optimizer state for momentum/adam. Not serialized for simplicity; reinitialized at load.
    #[serde(skip)]
    pub mom_w1: Option<Vec<f64>>,
    #[serde(skip)]
    pub mom_b1: Option<Vec<f64>>,
    #[serde(skip)]
    pub mom_w2: Option<Vec<f64>>,
    #[serde(skip)]
    pub mom_b2: Option<Vec<f64>>,
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
}

impl SimpleMlp {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // initialize small random-ish weights
    // use smaller initialization to reduce chance of exploding gradients
    let w1 = vec![0.001f64; input_dim * hidden_dim];
        let b1 = vec![0.0f64; hidden_dim];
    let w2 = vec![0.001f64; hidden_dim * output_dim];
        let b2 = vec![0.0f64; output_dim];
        Self {
            w1,
            b1,
            w2,
            b2,
            input_dim,
            hidden_dim,
            output_dim,
            update_count: 0u32,
            mom_w1: None,
            mom_b1: None,
            mom_w2: None,
            mom_b2: None,
            adam_m_w1: None,
            adam_v_w1: None,
            adam_m_b1: None,
            adam_v_b1: None,
            adam_m_w2: None,
            adam_v_w2: None,
            adam_m_b2: None,
            adam_v_b2: None,
        }
    }

    /// Initialize optimizer state (momentum/adam) to zero vectors sized to params.
    pub fn optimizer_init(&mut self) {
        let z_w1 = vec![0.0f64; self.w1.len()];
        let z_b1 = vec![0.0f64; self.b1.len()];
        let z_w2 = vec![0.0f64; self.w2.len()];
        let z_b2 = vec![0.0f64; self.b2.len()];
        self.mom_w1 = Some(z_w1.clone());
        self.mom_b1 = Some(z_b1.clone());
        self.mom_w2 = Some(z_w2.clone());
        self.mom_b2 = Some(z_b2.clone());
        self.adam_m_w1 = Some(z_w1.clone());
        self.adam_v_w1 = Some(z_w1.clone());
        self.adam_m_b1 = Some(z_b1.clone());
        self.adam_v_b1 = Some(z_b1.clone());
        self.adam_m_w2 = Some(z_w2.clone());
        self.adam_v_w2 = Some(z_w2.clone());
        self.adam_m_b2 = Some(z_b2.clone());
        self.adam_v_b2 = Some(z_b2.clone());
    }

    pub fn score(&self, features: &[f64]) -> Vec<f64> {
        // very small forward: features (input_dim) -> hidden -> relu -> output
        assert_eq!(features.len(), self.input_dim);
        // use f64 for accumulation, then cast outputs to f32
        let mut hidden = vec![0.0f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += (features[j] as f64) * self.w1[j * self.hidden_dim + i];
            }
            hidden[i] = if sum > 0.0f64 { sum } else { 0.0f64 };
        }
        let mut out = vec![0.0f64; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = self.b2[i];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[j * self.output_dim + i];
            }
            out[i] = sum; // raw score in f64
        }
        out
    }

    // Perform a single SGD step with mean-squared error on the output
    pub fn update(&mut self, features: &[f64], target: &[f64], lr: f64, weight_decay: f64, momentum: f64, optimizer: &str) {
        // single-sample update using f64 intermediates for numerical stability
    assert_eq!(features.len(), self.input_dim);
    assert_eq!(target.len(), self.output_dim);
    let lr_f64 = lr;

        // forward in f64
        let mut hidden = vec![0.0f64; self.hidden_dim];
        let mut hidden_pre = vec![0.0f64; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut sum = self.b1[i];
            for j in 0..self.input_dim {
                sum += features[j] * self.w1[j * self.hidden_dim + i];
            }
            hidden_pre[i] = sum;
            hidden[i] = if sum > 0.0f64 { sum } else { 0.0f64 };
        }
    let mut out = vec![0.0f64; self.output_dim];
        for i in 0..self.output_dim {
            let mut sum = self.b2[i];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[j * self.output_dim + i];
            }
            out[i] = sum;
        }

        // compute gradients (MSE) in f64 and clip
        let mut grad_out = vec![0.0f64; self.output_dim];
        let max_grad = 0.1f64;
        for i in 0..self.output_dim {
            let g = 2.0f64 * (out[i] - target[i]) / (self.output_dim as f64);
            grad_out[i] = if g.is_finite() {
                if g > max_grad { max_grad } else if g < -max_grad { -max_grad } else { g }
            } else { 0.0f64 };
        }

        // grads for w2 and b2 (apply directly to f64 params)
        for i in 0..self.output_dim {
            // b2
            let gb = grad_out[i];
            let mut gb_final = gb;
            // L2 weight decay on biases (rare, but keep consistent)
            if weight_decay > 0.0 { gb_final += weight_decay * self.b2[i]; }
            if optimizer == "adam" {
                // Adam update for b2
                if self.adam_m_b2.is_none() { self.optimizer_init(); }
                let m = self.adam_m_b2.as_mut().unwrap();
                let v = self.adam_v_b2.as_mut().unwrap();
                let idx = i;
                m[idx] = 0.9f64 * m[idx] + 0.1f64 * gb_final;
                v[idx] = 0.999f64 * v[idx] + 0.001f64 * gb_final * gb_final;
                let m_hat = m[idx] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[idx] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.b2[i] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_b2.is_none() { self.optimizer_init(); }
                let mom = self.mom_b2.as_mut().unwrap();
                mom[i] = momentum * mom[i] + (1.0 - momentum) * gb_final;
                self.b2[i] -= lr_f64 * mom[i];
            } else {
                self.b2[i] -= lr_f64 * gb_final;
            }
            for j in 0..self.hidden_dim {
                let idx = j * self.output_dim + i;
                let mut g = grad_out[i] * hidden[j];
                if !g.is_finite() { g = 0.0f64; }
                if g > max_grad { g = max_grad } else if g < -max_grad { g = -max_grad }
                // apply L2 weight decay
                if weight_decay > 0.0 { g += weight_decay * self.w2[idx]; }
                if optimizer == "adam" {
                    if self.adam_m_w2.is_none() { self.optimizer_init(); }
                    let m = self.adam_m_w2.as_mut().unwrap();
                    let v = self.adam_v_w2.as_mut().unwrap();
                    m[idx] = 0.9f64 * m[idx] + 0.1f64 * g;
                    v[idx] = 0.999f64 * v[idx] + 0.001f64 * g * g;
                    let m_hat = m[idx] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                    let v_hat = v[idx] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                    self.w2[idx] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
                } else if momentum > 0.0 {
                    if self.mom_w2.is_none() { self.optimizer_init(); }
                    let mom = self.mom_w2.as_mut().unwrap();
                    mom[idx] = momentum * mom[idx] + (1.0 - momentum) * g;
                    self.w2[idx] -= lr_f64 * mom[idx];
                } else {
                    self.w2[idx] -= lr_f64 * g;
                }
            }
        }

        // backprop into hidden (f64)
        let mut grad_hidden = vec![0.0f64; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = 0.0f64;
            for i in 0..self.output_dim {
                sum += grad_out[i] * self.w2[j * self.output_dim + i];
            }
            let deriv = if hidden_pre[j] > 0.0f64 { 1.0f64 } else { 0.0f64 };
            let mut gh = sum * deriv;
            if !gh.is_finite() { gh = 0.0f64; }
            if gh > max_grad { gh = max_grad } else if gh < -max_grad { gh = -max_grad }
            grad_hidden[j] = gh;
        }

        // grads for w1 and b1
        for j in 0..self.hidden_dim {
            let mut gb = grad_hidden[j];
            if !gb.is_finite() { gb = 0.0f64; }
            if gb > max_grad { gb = max_grad } else if gb < -max_grad { gb = -max_grad }
            // weight decay
            let mut gb_final = gb;
            if weight_decay > 0.0 { gb_final += weight_decay * self.b1[j]; }
            if optimizer == "adam" {
                if self.adam_m_b1.is_none() { self.optimizer_init(); }
                let m = self.adam_m_b1.as_mut().unwrap();
                let v = self.adam_v_b1.as_mut().unwrap();
                m[j] = 0.9f64 * m[j] + 0.1f64 * gb_final;
                v[j] = 0.999f64 * v[j] + 0.001f64 * gb_final * gb_final;
                let m_hat = m[j] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[j] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.b1[j] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_b1.is_none() { self.optimizer_init(); }
                let mom = self.mom_b1.as_mut().unwrap();
                mom[j] = momentum * mom[j] + (1.0 - momentum) * gb_final;
                self.b1[j] -= lr_f64 * mom[j];
            } else {
                self.b1[j] -= lr_f64 * gb_final;
            }
            for i in 0..self.input_dim {
                let idx = i * self.hidden_dim + j;
                let mut g = grad_hidden[j] * features[i];
                if !g.is_finite() { g = 0.0f64; }
                if g > max_grad { g = max_grad } else if g < -max_grad { g = -max_grad }
                if weight_decay > 0.0 { g += weight_decay * self.w1[idx]; }
                if optimizer == "adam" {
                    if self.adam_m_w1.is_none() { self.optimizer_init(); }
                    let m = self.adam_m_w1.as_mut().unwrap();
                    let v = self.adam_v_w1.as_mut().unwrap();
                    m[idx] = 0.9f64 * m[idx] + 0.1f64 * g;
                    v[idx] = 0.999f64 * v[idx] + 0.001f64 * g * g;
                    let m_hat = m[idx] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                    let v_hat = v[idx] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                    self.w1[idx] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
                } else if momentum > 0.0 {
                    if self.mom_w1.is_none() { self.optimizer_init(); }
                    let mom = self.mom_w1.as_mut().unwrap();
                    mom[idx] = momentum * mom[idx] + (1.0 - momentum) * g;
                    self.w1[idx] -= lr_f64 * mom[idx];
                } else {
                    self.w1[idx] -= lr_f64 * g;
                }
            }
        }

        // Basic NaN/Inf guard: if any parameter became non-finite, replace those entries with 0.0
    if !self.w1.iter().all(|x| x.is_finite()) || !self.b1.iter().all(|x| x.is_finite()) || !self.w2.iter().all(|x| x.is_finite()) || !self.b2.iter().all(|x| x.is_finite()) {
            eprintln!("SimpleMlp::update: detected non-finite parameter values (Inf/NaN). Clamping non-finite entries to 0.0");
            for x in &mut self.w1 { if !x.is_finite() { *x = 0.0f64 } }
            for x in &mut self.b1 { if !x.is_finite() { *x = 0.0f64 } }
            for x in &mut self.w2 { if !x.is_finite() { *x = 0.0f64 } }
            for x in &mut self.b2 { if !x.is_finite() { *x = 0.0f64 } }
        }
    self.update_count = self.update_count.saturating_add(1);
    }

    /// Apply an averaged SGD update over a minibatch of inputs/targets.
    /// Each input is &[f32] of length input_dim; targets are &[f32] of length output_dim.
    pub fn update_batch(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], lr: f64, weight_decay: f64, momentum: f64, optimizer: &str) {
        if inputs.is_empty() { return; }
        let batch_size = inputs.len() as f64;

        // We'll accumulate gradients across the batch using f64 for stability then apply the average update.
        // Initialize accumulators in f64
        let mut acc_w2 = vec![0.0f64; self.w2.len()];
        let mut acc_b2 = vec![0.0f64; self.b2.len()];
        let mut acc_w1 = vec![0.0f64; self.w1.len()];
        let mut acc_b1 = vec![0.0f64; self.b1.len()];

    for (x, y) in inputs.iter().zip(targets.iter()) {
            // forward using f64 accumulators for dot products
            let mut hidden = vec![0.0f64; self.hidden_dim];
            let mut hidden_pre = vec![0.0f64; self.hidden_dim];
            for i in 0..self.hidden_dim {
                let mut sum = self.b1[i] as f64;
                for j in 0..self.input_dim {
                    sum += (x[j] as f64) * (self.w1[j * self.hidden_dim + i] as f64);
                }
                hidden_pre[i] = sum;
                hidden[i] = if sum > 0.0f64 { sum } else { 0.0f64 };
            }
            let mut out = vec![0.0f64; self.output_dim];
            for i in 0..self.output_dim {
                let mut sum = self.b2[i];
                for j in 0..self.hidden_dim {
                    sum += hidden[j] * self.w2[j * self.output_dim + i];
                }
                out[i] = sum;
            }

            // grad out (MSE) in f64
            let mut grad_out = vec![0.0f64; self.output_dim];
            for i in 0..self.output_dim {
                grad_out[i] = 2.0f64 * (out[i] - (y[i] as f64)) / (self.output_dim as f64);
            }

            // accumulate grads for w2/b2
            for i in 0..self.output_dim {
                acc_b2[i] += grad_out[i];
                for j in 0..self.hidden_dim {
                    let idx = j * self.output_dim + i;
                    acc_w2[idx] += grad_out[i] * hidden[j];
                }
            }

            // backprop into hidden (f64)
            let mut grad_hidden = vec![0.0f64; self.hidden_dim];
            for j in 0..self.hidden_dim {
                let mut sum = 0.0f64;
                for i in 0..self.output_dim {
                    sum += grad_out[i] * self.w2[j * self.output_dim + i];
                }
                let deriv = if hidden_pre[j] > 0.0f64 { 1.0f64 } else { 0.0f64 };
                grad_hidden[j] = sum * deriv;
            }

            for j in 0..self.hidden_dim {
                acc_b1[j] += grad_hidden[j];
                for i in 0..self.input_dim {
                    let idx = i * self.hidden_dim + j;
                    acc_w1[idx] += grad_hidden[j] * x[i];
                }
            }
        }

    // Average accumulators and apply with lr (reuse clipping logic from update)
        let avg_w2 = acc_w2.iter().map(|v| v / batch_size).collect::<Vec<f64>>();
        let avg_b2 = acc_b2.iter().map(|v| v / batch_size).collect::<Vec<f64>>();
        let avg_w1 = acc_w1.iter().map(|v| v / batch_size).collect::<Vec<f64>>();
        let avg_b1 = acc_b1.iter().map(|v| v / batch_size).collect::<Vec<f64>>();

        // apply clipping and update
        let max_grad = 0.1f64;
    let lr_f64 = lr;
        // Apply updates with chosen optimizer, momentum, and weight decay
        for i in 0..avg_b2.len() {
            let mut gb = avg_b2[i]; if !gb.is_finite() { gb = 0.0 };
            if gb > max_grad { gb = max_grad } else if gb < -max_grad { gb = -max_grad }
            let mut gb_final = gb;
            if weight_decay > 0.0 { gb_final += weight_decay * self.b2[i]; }
            if optimizer == "adam" {
                if self.adam_m_b2.is_none() { self.optimizer_init(); }
                let m = self.adam_m_b2.as_mut().unwrap();
                let v = self.adam_v_b2.as_mut().unwrap();
                m[i] = 0.9f64 * m[i] + 0.1f64 * gb_final;
                v[i] = 0.999f64 * v[i] + 0.001f64 * gb_final * gb_final;
                let m_hat = m[i] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[i] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.b2[i] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_b2.is_none() { self.optimizer_init(); }
                let mom = self.mom_b2.as_mut().unwrap();
                mom[i] = momentum * mom[i] + (1.0 - momentum) * gb_final;
                self.b2[i] -= lr_f64 * mom[i];
            } else {
                self.b2[i] -= lr_f64 * gb_final;
            }
        }
        for i in 0..avg_w2.len() {
            let mut gw = avg_w2[i]; if !gw.is_finite() { gw = 0.0 };
            if gw > max_grad { gw = max_grad } else if gw < -max_grad { gw = -max_grad }
            if weight_decay > 0.0 { gw += weight_decay * self.w2[i]; }
            if optimizer == "adam" {
                if self.adam_m_w2.is_none() { self.optimizer_init(); }
                let m = self.adam_m_w2.as_mut().unwrap();
                let v = self.adam_v_w2.as_mut().unwrap();
                m[i] = 0.9f64 * m[i] + 0.1f64 * gw;
                v[i] = 0.999f64 * v[i] + 0.001f64 * gw * gw;
                let m_hat = m[i] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[i] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.w2[i] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_w2.is_none() { self.optimizer_init(); }
                let mom = self.mom_w2.as_mut().unwrap();
                mom[i] = momentum * mom[i] + (1.0 - momentum) * gw;
                self.w2[i] -= lr_f64 * mom[i];
            } else {
                self.w2[i] -= lr_f64 * gw;
            }
        }
        for i in 0..avg_b1.len() {
            let mut gb = avg_b1[i]; if !gb.is_finite() { gb = 0.0 };
            if gb > max_grad { gb = max_grad } else if gb < -max_grad { gb = -max_grad }
            if weight_decay > 0.0 { gb += weight_decay * self.b1[i]; }
            if optimizer == "adam" {
                if self.adam_m_b1.is_none() { self.optimizer_init(); }
                let m = self.adam_m_b1.as_mut().unwrap();
                let v = self.adam_v_b1.as_mut().unwrap();
                m[i] = 0.9f64 * m[i] + 0.1f64 * gb;
                v[i] = 0.999f64 * v[i] + 0.001f64 * gb * gb;
                let m_hat = m[i] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[i] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.b1[i] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_b1.is_none() { self.optimizer_init(); }
                let mom = self.mom_b1.as_mut().unwrap();
                mom[i] = momentum * mom[i] + (1.0 - momentum) * gb;
                self.b1[i] -= lr_f64 * mom[i];
            } else {
                self.b1[i] -= lr_f64 * gb;
            }
        }
        for i in 0..avg_w1.len() {
            let mut gw = avg_w1[i]; if !gw.is_finite() { gw = 0.0 };
            if gw > max_grad { gw = max_grad } else if gw < -max_grad { gw = -max_grad }
            if weight_decay > 0.0 { gw += weight_decay * self.w1[i]; }
            if optimizer == "adam" {
                if self.adam_m_w1.is_none() { self.optimizer_init(); }
                let m = self.adam_m_w1.as_mut().unwrap();
                let v = self.adam_v_w1.as_mut().unwrap();
                m[i] = 0.9f64 * m[i] + 0.1f64 * gw;
                v[i] = 0.999f64 * v[i] + 0.001f64 * gw * gw;
                let m_hat = m[i] / (1.0 - 0.9f64.powi(self.update_count as i32 + 1));
                let v_hat = v[i] / (1.0 - 0.999f64.powi(self.update_count as i32 + 1));
                self.w1[i] -= lr_f64 * m_hat / (v_hat.sqrt() + 1e-8f64);
            } else if momentum > 0.0 {
                if self.mom_w1.is_none() { self.optimizer_init(); }
                let mom = self.mom_w1.as_mut().unwrap();
                mom[i] = momentum * mom[i] + (1.0 - momentum) * gw;
                self.w1[i] -= lr_f64 * mom[i];
            } else {
                self.w1[i] -= lr_f64 * gw;
            }
        }

        // guard against NaN/Inf
    if !self.w1.iter().all(|x| x.is_finite()) || !self.b1.iter().all(|x| x.is_finite()) || !self.w2.iter().all(|x| x.is_finite()) || !self.b2.iter().all(|x| x.is_finite()) {
            eprintln!("SimpleMlp::update_batch: detected non-finite parameter values (Inf/NaN). Clamping non-finite entries to 0.0");
            for x in &mut self.w1 { if !x.is_finite() { *x = 0.0 } }
            for x in &mut self.b1 { if !x.is_finite() { *x = 0.0 } }
            for x in &mut self.w2 { if !x.is_finite() { *x = 0.0 } }
            for x in &mut self.b2 { if !x.is_finite() { *x = 0.0 } }
        }
    self.update_count = self.update_count.saturating_add(1);
    }

    pub fn save(&self, path: &PathBuf) -> Result<(), std::io::Error> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let s = serde_yaml::to_string(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        fs::write(path, s)
    }

    /// Save pretty-printed YAML (same as save but kept for callers expecting pretty)
    pub fn save_pretty(&self, path: &PathBuf) -> Result<(), std::io::Error> {
        // serde_yaml::to_string already produces human-readable YAML
        self.save(path)
    }

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
    fn simple_forward_and_update() {
        let mut m = SimpleMlp::new(3, 4, 1);
    let features = [0.1f64, 0.2, 0.3];
    let before = m.score(&features);
    m.update(&features, &[1.0], 0.01, 1e-4, 0.9, "sgd");
    let after = m.score(&features);
    // score should change after update and params should be finite
    assert_ne!(before, after);
    assert!(m.w1.iter().all(|x| x.is_finite()));
    assert!(m.w2.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");
        let m = SimpleMlp::new(2, 2, 1);
        m.save(&path).unwrap();
        let m2 = SimpleMlp::load(&path).unwrap();
        assert_eq!(m.input_dim, m2.input_dim);
        assert_eq!(m.hidden_dim, m2.hidden_dim);
        assert_eq!(m.output_dim, m2.output_dim);
    }

    #[test]
    fn batch_update_stability() {
        let mut m = SimpleMlp::new(3, 4, 1);
        let inputs = vec![vec![0.1f64, 0.2, 0.3]; 8];
        let targets = vec![vec![1.0f64]; 8];
        let before = m.score(&inputs[0]);
    m.update_batch(&inputs, &targets, 0.01, 1e-4, 0.9, "sgd");
        let after = m.score(&inputs[0]);
        assert_ne!(before, after);
        assert!(m.w1.iter().all(|x| x.is_finite()));
        assert!(m.w2.iter().all(|x| x.is_finite()));
    }
}
