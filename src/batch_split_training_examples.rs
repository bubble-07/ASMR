use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};
use std::collections::HashMap;
use rand::Rng;
use std::ops::Range;
use crate::training_examples::*;
use crate::batch_split::*;

pub struct BatchSplitPlayoutBundle {
    pub playout_bundle : PlayoutBundle,
    pub weight : f64,
    pub batch_split : BatchSplit,
}
pub struct BatchSplitTrainingExamples {
    pub training_examples : HashMap<(usize, usize), BatchSplitPlayoutBundle>,
}

impl BatchSplitTrainingExamples {
    ///Iterates over one round of training batches (one training batch per (set-size, game-length)
    ///pair). All of the weights in the sequence sum to 1.
    pub fn iter_training_batches<'a, R : Rng + ?Sized>(&'a self, rng : &'a mut R, device : Device) -> 
                                impl Iterator<Item = (f64, PlayoutBundle)> + 'a {
        self.training_examples.values()
            .filter_map(move |batch_split_bundle| {
                let batch_index_range = batch_split_bundle.batch_split.grab_training_batch(rng)?;
                let weight = batch_split_bundle.weight; 
                let playout_bundle_batch = batch_split_bundle.playout_bundle.grab_batch(batch_index_range, device);
                Option::Some((weight, playout_bundle_batch))
            })
    }
    ///Iterates over all validation batches. All of the weights in the sequence sum to 1.
    pub fn iter_validation_batches<'a>(&'a self, device : Device) -> 
                                impl Iterator<Item = (f64, PlayoutBundle)> + 'a {
        self.training_examples.values()
            .flat_map(move |batch_split_bundle| {
                let validation_batches = batch_split_bundle.batch_split.iter_validation_batches();
                validation_batches.map(move |(batch_weight, batch_index_range)| {
                    let weight = batch_split_bundle.weight * batch_weight;
                    let playout_bundle_batch = batch_split_bundle.playout_bundle.grab_batch(batch_index_range, device);
                    (weight, playout_bundle_batch)
                })
            })
    }

    pub fn from_training_examples(mut training_examples : TrainingExamples,
                                  batch_size : usize,
                                  recommended_min_validation_batches : usize) -> Self {
        let mut total_num_examples : usize = 0;
        for playout_bundle in training_examples.playout_bundles.values() {
            total_num_examples += playout_bundle.get_num_playouts();
        }
        let total_num_examples = total_num_examples;

        let mut result = HashMap::new();
        for (key, playout_bundle) in training_examples.playout_bundles.drain() {
            let full_size = playout_bundle.get_num_playouts();
            let batch_split = BatchSplit::new(full_size, batch_size, recommended_min_validation_batches);
            let weight = (full_size as f64) / (total_num_examples as f64);

            let playout_bundle = BatchSplitPlayoutBundle {
                playout_bundle,
                batch_split,
                weight,
            };

            result.insert(key, playout_bundle);
        }
        BatchSplitTrainingExamples {
            training_examples : result,
        }
    }
}
