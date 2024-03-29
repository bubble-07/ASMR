use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};
use std::collections::HashMap;
use rand::Rng;
use std::ops::Range;
use std::iter::zip;
use crate::playout_sketches::*;
use crate::training_examples::*;
use crate::batch_split::*;
use crate::params::*;

pub struct BatchSplitPlayoutBundle<BundleType : PlayoutBundleLike> {
    pub playout_bundle : BundleType,
    pub weight : f64,
    pub batch_split : BatchSplit,
}
pub struct BatchSplitTrainingExamples<BundleType : PlayoutBundleLike> {
    pub training_examples : HashMap<(usize, usize), BatchSplitPlayoutBundle<BundleType>>,
}

struct ValidationBatchIterator<'a, BundleType : PlayoutBundleLike> {
    batch_split_training_examples : &'a BatchSplitTrainingExamples<BundleType>,
    device : Device,
    plan : Vec<Vec<((usize, usize), f64, Range<i64>)>>,
}
impl <'a, BundleType : PlayoutBundleLike> Iterator for ValidationBatchIterator<'a, BundleType> {
    type Item = Vec<(f64, BundleType)>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut step = self.plan.pop()?;
        let device = self.device;
        let training_examples = &self.batch_split_training_examples.training_examples;
        let step_result : Vec<(f64, BundleType)> = 
            step.drain(..).map(|(key, weight, range)| {
                let playout_bundle = training_examples.get(&key).unwrap().playout_bundle.grab_batch(range, device);
                (weight, playout_bundle)
            }).collect();
        Option::Some(step_result)
    }
}

impl BatchSplitTrainingExamples<PlayoutSketchBundle> {
    pub fn iter_playout_bundle_training_batches<'a, R : Rng + ?Sized>(&'a self, params : &'a Params, 
                                                                      rng : &'a mut R) ->
                                impl Iterator<Item = (f64, PlayoutBundle)> + 'a {
        let device = params.get_device();

        //Step 1: Draw batches and reorganize to bunch together bundles with the same starting set size
        let mut batch_bundles_by_starting_set_size = HashMap::new();
        for (key, full_batch_split_sketches) in self.training_examples.iter() {
            let (init_set_size, _) = key;
            let maybe_batch_index_range = full_batch_split_sketches.batch_split.grab_training_batch(rng);
            if maybe_batch_index_range.is_none() {
                continue;
            }
            let batch_index_range = maybe_batch_index_range.unwrap();
            let weight = full_batch_split_sketches.weight; 
            let playout_sketch_bundle_batch = full_batch_split_sketches.playout_bundle.grab_batch(batch_index_range, device);

            if !batch_bundles_by_starting_set_size.contains_key(&init_set_size) {
                batch_bundles_by_starting_set_size.insert(init_set_size, Vec::new());
            }
            let entries = batch_bundles_by_starting_set_size.get_mut(&init_set_size).unwrap();
            entries.push((weight, playout_sketch_bundle_batch));
        }

        //Step 2: Loop through the different initial set sizes and generate matrices
        batch_bundles_by_starting_set_size.into_iter().flat_map(move |(init_set_size, entries)| {
            let num_matrices_by_entry : Vec<_> = entries.iter()
                                           .map(|(_, playout_sketch_bundle)| {
                                               (init_set_size * playout_sketch_bundle.get_num_playouts()) as i64
                                            })
                                           .collect();
            let total_num_matrices = num_matrices_by_entry.iter().sum::<i64>() as usize;
            let random_matrices = params.generate_random_matrices(total_num_matrices);
            let random_matrices_split = random_matrices.split_with_sizes(num_matrices_by_entry.as_slice(), 0);

            zip(entries.into_iter(), random_matrices_split)
            .map(|(entry, random_matrices)| {
                let (weight, playout_sketch_bundle) = entry;
                let playout_bundle = PlayoutBundle::from_initial_matrices_and_sketch_bundle(random_matrices, playout_sketch_bundle);
                (weight, playout_bundle)
            })
        })
    }
}

impl <BundleType : PlayoutBundleLike> BatchSplitTrainingExamples<BundleType> {
    ///Iterates over one round of training batches (one training batch per (set-size, game-length)
    ///pair). All of the weights in the sequence sum to 1.
    pub fn iter_training_batches<'a, R : Rng + ?Sized>(&'a self, rng : &'a mut R, device : Device) -> 
                                impl Iterator<Item = (f64, BundleType)> + 'a {
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
                                impl Iterator<Item = Vec<(f64, BundleType)>> + 'a {

        //First, collect all of the validation batches for all of the training examples.
        let mut validation_batch_indices_and_weights : 
            HashMap<(usize, usize), Vec<(f64, Range<i64>)>> = 
                                       self.training_examples.iter()
                                       .map(|(key, batch_split_bundle)| {
                                           let result_vec = batch_split_bundle.batch_split.iter_validation_batches()
                                                                              .collect();
                                           (*key, result_vec)
                                        }).collect();
        //Find the largest number of validation batches for any key
        let max_batches_per_key = validation_batch_indices_and_weights.values()
                                  .map(|vec| vec.len())
                                  .max().unwrap();

        //Build a (key, weight, range) doubly-nested vec first, where
        //the inner dimension traverses different buckets, and the outer
        //dimension is just repeated occurrences.
        let mut plan = Vec::new();
        for _ in 0..max_batches_per_key {
            plan.push(Vec::new());
        }
        for key in self.training_examples.keys() {
            let bundle_weight = self.training_examples.get(key).unwrap().weight;
            let indices_and_weights = validation_batch_indices_and_weights.get_mut(key).unwrap();
            for (combined_validation_batch_num, (batch_weight, indices)) in indices_and_weights.drain(..)
                                                                      .enumerate() {
                let weight = bundle_weight * batch_weight;
                plan[combined_validation_batch_num].push((*key, weight, indices));
            }
        }
        //With the plan in place, finally return the requested iterator.
        let result = ValidationBatchIterator {
            batch_split_training_examples : &self,
            device,
            plan
        };
        result
    }

    pub fn from_training_examples(mut training_examples : TrainingExamples<BundleType>,
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
