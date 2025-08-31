use crate::history::History;
use crate::settings::Settings;
use crate::training_sample_generator::TrainingSampleGenerator;
use crate::ml::scalable_mlp::ScalableMlp;

#[derive(Debug)]
pub struct Trainer<'a> {
    settings: &'a Settings,
    history: &'a mut History,
}

impl<'a> Trainer<'a> {
    pub fn new(settings: &'a Settings, history: &'a mut History) -> Trainer<'a> {
        Trainer { settings, history }
    }

    /// Offline trainer that loads/generates a dataset and performs several
    /// epochs of SGD on a ScalableMlp model persisted at `scalable-ml.yaml`.
    ///
    /// - epochs: number of passes over the dataset (default 5 if 0)
    /// - records: optionally limit number of records per epoch
    pub fn train(&mut self, epochs: usize, records: Option<usize>) {
        let epochs = if epochs == 0 { self.settings.trainer_epochs } else { epochs };
        let lr = self.settings.learning_rate as f64;

        // Create or load the dataset (will cache to disk internally).
        let generator = TrainingSampleGenerator::new(self.settings, &self.history);

        // Determine model path and load or create model.
        let model_path = match Settings::mcfly_db_path().parent() {
            Some(p) => p.join("scalable-ml.yaml"),
            None => {
                eprintln!("Trainer: unable to determine model path");
                return;
            }
        };

        // Input dim corresponds to number of feature fields (16 now with enhanced features)
        let input_dim = 16usize;
        let hidden_dim = self.settings.trainer_hidden_dim;
        let output_dim = 1usize;

        let mut model = if let Some(mut m) = ScalableMlp::load(&model_path) {
            m.optimizer_init();
            m
        } else {
            // Create a 3-layer network for better capacity
            let mut m = ScalableMlp::new(input_dim, hidden_dim, hidden_dim / 2, output_dim);
            m.optimizer_init();
            m
        };

        println!("Trainer: starting offline training (epochs={epochs})");

        let batch_size = self.settings.trainer_batch_size;

        for epoch in 0..epochs {
            let mut seen: usize = 0;

            // Simple step LR scheduler: decay every lr_decay_step epochs
            let mut effective_lr = lr;
            if let Some(step) = self.settings.lr_decay_step {
                if step > 0 && epoch > 0 && (epoch % step) == 0 {
                    effective_lr = effective_lr * self.settings.lr_decay_factor;
                }
            }

            if let Some(bs) = batch_size {
                // Collect batches then apply updates per example (simple approach)
                let mut batch_inputs: Vec<Vec<f64>> = Vec::with_capacity(bs);
                let mut batch_targets: Vec<Vec<f64>> = Vec::with_capacity(bs);

                generator.generate(records, |features, correct| {
                    // produce normalized input using the generator's stats
                    let input = generator.normalize_features(features);
                    let target = if correct { vec![1.0f64] } else { vec![0.0f64] };
                    batch_inputs.push(input);
                    batch_targets.push(target);
                    if batch_inputs.len() >= bs {
                        // apply averaged minibatch update
                        model.update_batch(&batch_inputs, &batch_targets, effective_lr, self.settings.weight_decay);
                        seen += batch_inputs.len();
                        batch_inputs.clear();
                        batch_targets.clear();
                    }
                });

                // flush remaining
                if !batch_inputs.is_empty() {
                    model.update_batch(&batch_inputs, &batch_targets, effective_lr, self.settings.weight_decay);
                    seen += batch_inputs.len();
                    batch_inputs.clear();
                    batch_targets.clear();
                }
            } else {
                generator.generate(records, |features, correct| {
                    let input = generator.normalize_features(features);
                    let target = if correct { vec![1.0f64] } else { vec![0.0f64] };
                    model.update(&input, &target, effective_lr, self.settings.weight_decay);
                    seen += 1;
                });
            }

            println!("Trainer: finished epoch {}/{} (processed {seen} samples)", epoch + 1, epochs);
        }

        if let Err(e) = model.save(&model_path) {
            eprintln!("Trainer: failed to save model: {e}");
        } else {
            println!("Trainer: saved trained model to {:?}", model_path);
        }
    }
}

