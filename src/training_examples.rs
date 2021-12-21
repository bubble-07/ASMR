use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use crate::game_data::*;
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;
use ndarray::*;
use crate::array_utils::*;
use std::convert::{TryFrom, TryInto};
use std::path::Path;
use crate::params::*;
use crate::turn_data::*;
use crate::network_config::*;

pub struct TrainingExamples {
    ///Mapping from set size to a K-element list of tensors of dims NxM,
    ///where K is the set size, N is the number of examples,
    ///and M is the flattened matrix dimension
    pub flattened_matrix_sets : HashMap<usize, Vec<Tensor>>,
    ///Mapping from set size to a tensor of dims NxM
    pub flattened_matrix_targets : HashMap<usize, Tensor>,
    ///Mapping from set size to a tensor of dims Nx(K*K)
    pub child_visit_probabilities : HashMap<usize, Tensor>
}

pub struct TrainingExamplesBuilder {
    ///K elements, axes are NxM
    pub flattened_matrix_sets : HashMap<usize, Vec<Vec<f32>>>,
    ///axes are NxM
    pub flattened_matrix_targets : HashMap<usize, Vec<f32>>,
    ///axes are NxKxK
    pub child_visit_probabilities : HashMap<usize, Vec<f32>>,
    ///Maps from the set size to the number of samples for that set size
    pub num_samples : HashMap<usize, usize>,
    pub m : usize
}

pub struct BatchIndex {
    pub set_sizing : usize,
    pub batch_index : usize
}

impl BatchIndex {
    pub fn get_normalized_loss(&self, params : &Params, training_examples : &TrainingExamples,
                               network_config : &NetworkConfig, 
                               device : Device) -> Tensor {
        let flattened_matrix_sets = training_examples.flattened_matrix_sets.get(&self.set_sizing).unwrap();
        let flattened_matrix_targets = training_examples.flattened_matrix_targets.get(&self.set_sizing).unwrap();
        let child_visit_probabilities = training_examples.child_visit_probabilities.get(&self.set_sizing).unwrap();

        let n = flattened_matrix_targets.size()[0] as usize;

        let start = (self.batch_index * params.batch_size) as i64;
        let size = std::cmp::min(params.batch_size as i64, (n as i64) - start);


        let matrix_sets_slices : Vec<Tensor> = flattened_matrix_sets.iter().map(|x|
                                  x.i(start..start + size).to_device(device).detach())
                                                             .collect();
        let matrix_targets_slices = flattened_matrix_targets.i(start..start + size)
                                    .to_device(device).detach();
        let child_visit_probabilities_slices = child_visit_probabilities.i(start..start + size)
                                    .to_device(device).detach();


        
        let normalized_loss = network_config.get_loss_from_scratch(&matrix_sets_slices, &matrix_targets_slices,
                                         &child_visit_probabilities_slices);
        normalized_loss
    }
}

impl TrainingExamples {
    pub fn get_validation_loss_weightings(&self, _params : &Params, set_sizings : &[usize]) -> Vec<f32> {
        let mut total_n = 0;
        let mut result = Vec::new();
        for set_sizing in set_sizings.iter() {
            let n = self.get_total_number_of_examples(*set_sizing);
            result.push(n as f32);
            total_n += n;
        }
        for i in 0..result.len() {
            result[i] /= total_n as f32;
        }
        result
    }
    pub fn get_training_loss_weightings(&self, params : &Params, set_sizings : &[usize]) -> Vec<f32> {
        let mut total_batches = 0;
        let mut result = Vec::new();
        for set_sizing in set_sizings.iter() {
            let batches = self.get_total_number_of_training_batches(params, *set_sizing);
            result.push(batches as f32);
            total_batches += batches;
        }
        for i in 0..result.len() {
            result[i] /= total_batches as f32;
        }
        result
    }
    pub fn get_set_sizings(&self) -> Vec<usize> {
        let set_sizings : Vec<usize> = self.flattened_matrix_sets.keys().map(|x| *x).collect();
        set_sizings
    }
    pub fn get_normalized_random_choice_loss(&self, set_sizing : usize) -> f64 {
        let target_policy = self.child_visit_probabilities.get(&set_sizing).unwrap();
        let k_squared = target_policy.size()[1];
        
        let nat_log = (k_squared as f64).ln();
        nat_log
    }
    pub fn get_total_number_of_validation_batches(&self, params : &Params, set_sizing : usize) -> usize {
        self.get_total_number_of_batches(params, set_sizing) - 
            self.get_total_number_of_training_batches(params, set_sizing)
    }
    pub fn get_total_number_of_training_batches(&self, params : &Params, set_sizing : usize) -> usize {
        let num_batches = self.get_total_number_of_batches(params, set_sizing);
        if (num_batches <= params.held_out_validation_batches) {
            0
        } else {
            num_batches - params.held_out_validation_batches
        }
    }
    pub fn get_total_number_of_batches(&self, params : &Params, set_sizing : usize) -> usize {
        let n = self.get_total_number_of_examples(set_sizing);
        let total_full_batches = n / params.batch_size;
        let has_partial_batch_at_end = n % params.batch_size != 0;
        if (has_partial_batch_at_end) {
            total_full_batches + 1
        } else {
            total_full_batches
        }
    }
    pub fn get_total_number_of_examples(&self, set_sizing : usize) -> usize {
        let n = self.flattened_matrix_targets.get(&set_sizing).unwrap().size()[0] as usize;
        n
    }

    fn concat_consume(a : Tensor, b : Tensor) -> Tensor {
        let result = Tensor::cat(&[a, b], 0);
        result
    }
    pub fn merge(&mut self, mut other : TrainingExamples) {
        let _guard = no_grad_guard();
        let mut set_sizes : Vec<usize> = other.flattened_matrix_sets.keys().map(|x| *x).collect();
        for set_size in set_sizes.drain(..) {
            let mut other_flattened_matrix_sets = other.flattened_matrix_sets.remove(&set_size).unwrap();
            let other_flattened_matrix_targets = other.flattened_matrix_targets.remove(&set_size).unwrap();
            let other_child_visit_probabilities = other.child_visit_probabilities.remove(&set_size).unwrap();
            //First, determine if we already have the set size.
            //If so, we'll append, but otherwise, we'll move
            if (self.flattened_matrix_sets.contains_key(&set_size)) {
                //Need to append the contents from the other to this one
                let mut my_flattened_matrix_sets = self.flattened_matrix_sets.remove(&set_size).unwrap();
                let my_flattened_matrix_targets = self.flattened_matrix_targets.remove(&set_size).unwrap();
                let my_child_visit_probabilities = self.child_visit_probabilities.remove(&set_size).unwrap();

                let flattened_matrix_targets = Self::concat_consume(my_flattened_matrix_targets,
                                                                    other_flattened_matrix_targets);
                let child_visit_probabilities = Self::concat_consume(my_child_visit_probabilities,
                                                                     other_child_visit_probabilities);
                let mut flattened_matrix_sets = Vec::with_capacity(set_size);
                for (my_flattened_matrix_set, other_flattened_matrix_set) in
                    my_flattened_matrix_sets.drain(..).zip(other_flattened_matrix_sets.drain(..)) {

                    let flattened_matrix_set = Self::concat_consume(my_flattened_matrix_set, 
                                                                    other_flattened_matrix_set);
                    flattened_matrix_sets.push(flattened_matrix_set);
                }

                self.flattened_matrix_sets.insert(set_size, flattened_matrix_sets);
                self.flattened_matrix_targets.insert(set_size, flattened_matrix_targets);
                self.child_visit_probabilities.insert(set_size, child_visit_probabilities);
            } else {
                //Need to move the contents from other into this one
                self.flattened_matrix_sets.insert(set_size, other_flattened_matrix_sets);
                self.flattened_matrix_targets.insert(set_size, other_flattened_matrix_targets);
                self.child_visit_probabilities.insert(set_size, other_child_visit_probabilities);
            }
        }
    }
}

impl TrainingExamplesBuilder {
    pub fn new(params : &Params) -> TrainingExamplesBuilder {
        let flattened_matrix_sets = HashMap::new();
        let flattened_matrix_targets = HashMap::new();
        let child_visit_probabilities = HashMap::new();
        let num_samples = HashMap::new();
        TrainingExamplesBuilder {
            flattened_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            num_samples,
            m : params.get_flattened_matrix_dim() as usize
        }
    }
    pub fn build<R : Rng + ?Sized>(mut self, rng : &mut R) -> TrainingExamples {
        let mut flattened_matrix_sets = HashMap::new();
        let mut flattened_matrix_targets = HashMap::new();
        let mut child_visit_probabilities = HashMap::new();

        let mut set_sizes : Vec<usize> = self.flattened_matrix_sets.keys().map(|x| *x).collect();

        for set_size in set_sizes.drain(..) {
            let k = set_size;

            let n = self.num_samples.remove(&set_size).unwrap();

            let mut prior_matrix_sets_vec = self.flattened_matrix_sets.remove(&set_size).unwrap();
            let mut matrix_sets_vec = Vec::new();
            for matrix_set in prior_matrix_sets_vec.drain(..) {
                let full = Array::from_vec(matrix_set).into_shape((n, self.m)).unwrap();
                matrix_sets_vec.push(full);
            }

            let matrix_targets = self.flattened_matrix_targets.remove(&set_size).unwrap();
            let visit_probabilities = self.child_visit_probabilities.remove(&set_size).unwrap();
            
            let matrix_targets = Array::from_vec(matrix_targets);
            let visit_probabilities = Array::from_vec(visit_probabilities);

            let mut matrix_targets = matrix_targets.into_shape((n, self.m)).unwrap();
            let mut visit_probabilities = visit_probabilities.into_shape((n, k * k)).unwrap();
            
            //Fisher-Yates shuffle the above three
            if (n > 1) {
                for i in 0..n-1 {
                    let j = rng.gen_range(i..n);

                    for z in 0..k {
                        swap_rows(&mut matrix_sets_vec[z], i, j);
                    }
                    swap_rows(&mut matrix_targets, i, j);
                    swap_rows(&mut visit_probabilities, i, j);
                }
            }

            let matrix_targets = matrix_targets.into_shape((n, self.m)).unwrap();
            let visit_probabilities = visit_probabilities.into_shape((n, k, k)).unwrap();

            let mut matrix_sets_tensors = Vec::new();
            for matrix_set in matrix_sets_vec.drain(..) {
                let flat_matrix_set = matrix_set.as_slice().unwrap();
                let mut matrix_set_tensor = Tensor::try_from(flat_matrix_set).unwrap();
                matrix_set_tensor = matrix_set_tensor.reshape(&[n as i64, self.m as i64]);
                matrix_sets_tensors.push(matrix_set_tensor);
            }
            flattened_matrix_sets.insert(set_size, matrix_sets_tensors);


            let flat_matrix_targets = matrix_targets.as_slice().unwrap();
            let flat_visit_probabilities = visit_probabilities.as_slice().unwrap();

            let mut matrix_targets_tensor = Tensor::try_from(flat_matrix_targets).unwrap();
            let mut visit_probabilities_tensor = Tensor::try_from(flat_visit_probabilities).unwrap();

            matrix_targets_tensor = matrix_targets_tensor.reshape(&[n as i64, self.m as i64]);
            visit_probabilities_tensor = visit_probabilities_tensor.reshape(&[n as i64, (k * k) as i64]);

            flattened_matrix_targets.insert(set_size, matrix_targets_tensor);
            child_visit_probabilities.insert(set_size, visit_probabilities_tensor);
        }
        TrainingExamples {
            flattened_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities
        }
    }

    pub fn add_game_data<R : Rng + ?Sized>(&mut self, game_data : GameData, rng : &mut R) {
        let mut turn_data_vec = game_data.get_turn_data();
        for turn_data in turn_data_vec.drain(..) {
            let permuted_turn_data = turn_data.permute(rng);
            self.add_turn_data(permuted_turn_data);
        }
    }

    pub fn add_turn_data(&mut self, mut turn_data : TurnData) {

        let set_size = turn_data.flattened_matrix_set.len();
        if (!self.flattened_matrix_sets.contains_key(&set_size)) {
            let mut matrix_set_vec = Vec::new();
            for _ in 0..set_size {
                matrix_set_vec.push(Vec::new());
            }
            self.flattened_matrix_sets.insert(set_size, matrix_set_vec);

            self.flattened_matrix_targets.insert(set_size, Vec::new());
            self.child_visit_probabilities.insert(set_size, Vec::new());
            self.num_samples.insert(set_size, 0);
        }

        let matrix_set_dest = self.flattened_matrix_sets.get_mut(&set_size).unwrap();
        for (flattened_matrix, i) in turn_data.flattened_matrix_set.drain(..).zip(0..set_size) {
            matrix_set_dest[i].extend_from_slice(flattened_matrix.as_slice().unwrap());
        }

        self.flattened_matrix_targets.get_mut(&set_size).unwrap()
            .extend_from_slice(turn_data.flattened_matrix_target.as_slice().unwrap());

        self.child_visit_probabilities.get_mut(&set_size).unwrap()
            .extend_from_slice(turn_data.child_visit_probabilities.as_slice().unwrap());

        let prev_num_samples = self.num_samples.remove(&set_size).unwrap(); 
        self.num_samples.insert(set_size, prev_num_samples + 1);
    }

}

impl TrainingExamples {
    pub fn save<T : AsRef<Path>>(mut self, path : T) -> Result<(), String> {
        let mut sizings = Vec::new();
        for sizing in self.flattened_matrix_sets.keys() {
            sizings.push(*sizing as i64);
        }
        let sizings = Tensor::try_from(sizings.as_slice()).unwrap();

        let mut named_tensors = Vec::new();

        named_tensors.push((String::from("sizings"), sizings));

        for (sizing, mut tensor_vec) in self.flattened_matrix_sets.drain() {
            for (tensor, i) in tensor_vec.drain(..).zip(0..sizing) {
                let name = format!("{}flattened_matrix_sets{}", i, sizing);
                named_tensors.push((name, tensor));
            }
        }
        for (sizing, tensor) in self.flattened_matrix_targets.drain() {
            let name = format!("flattened_matrix_targets{}", sizing);
            named_tensors.push((name, tensor));
        }
        for (sizing, tensor) in self.child_visit_probabilities.drain() {
            let name = format!("child_visit_probabilities{}", sizing);
            named_tensors.push((name, tensor));
        }
        let maybe_save_result = Tensor::save_multi(&named_tensors, path);
        match (maybe_save_result) {
            Result::Ok(_) => Result::Ok(()),
            Result::Err(err) => Result::Err(format!("Could not save training examples: {}", err))
        }
    }
    pub fn load<T : AsRef<Path>>(path : T, device : Device) -> Result<TrainingExamples, String> {
        let maybe_named_tensors = Tensor::load_multi_with_device(path, device);
        match (maybe_named_tensors) {
            Result::Ok(mut named_tensors) => {
                let mut flattened_matrix_sets = HashMap::new();
                let mut flattened_matrix_targets = HashMap::new();
                let mut child_visit_probabilities = HashMap::new();

                let mut named_tensor_map = HashMap::new(); 
                for (name, tensor) in named_tensors.drain(..) {
                    named_tensor_map.insert(name, tensor.detach());
                }
                let sizings = named_tensor_map.remove("sizings").unwrap();
                let num_sizings = sizings.size()[0];
                for i in 0..num_sizings {
                    let sizing = sizings.int64_value(&[i as i64]) as usize;

                    let mut flattened_matrix_set_vec = Vec::new();
                    for j in 0..sizing {
                        let flattened_matrix_set_name = format!("{}flattened_matrix_sets{}", j, sizing);
                        let flattened_matrix_set = named_tensor_map.remove(&flattened_matrix_set_name).unwrap();
                        flattened_matrix_set_vec.push(flattened_matrix_set);
                    }
                    flattened_matrix_sets.insert(sizing, flattened_matrix_set_vec);


                    let flattened_matrix_target_name = format!("flattened_matrix_targets{}", sizing);
                    let child_visit_probability_name = format!("child_visit_probabilities{}", sizing);

                                        let flattened_matrix_target = named_tensor_map.remove(&flattened_matrix_target_name).unwrap();
                    let child_visit_probability = named_tensor_map.remove(&child_visit_probability_name).unwrap();

                    flattened_matrix_targets.insert(sizing, flattened_matrix_target);
                    child_visit_probabilities.insert(sizing, child_visit_probability);

                }
                Result::Ok(TrainingExamples {
                    flattened_matrix_sets,
                    flattened_matrix_targets,
                    child_visit_probabilities
                })
            },
            Result::Err(err) => {
                Result::Err(format!("Could not load training examples: {}", err))
            }
        }
    }
}
