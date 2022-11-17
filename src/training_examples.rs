use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};
use std::collections::HashMap;
use rand::Rng;
use std::ops::Range;
use rand::seq::SliceRandom;
use ndarray::*;
use crate::array_utils::*;
use std::convert::{TryFrom, TryInto};
use std::path::Path;
use crate::params::*;
use crate::network_config::*;
use crate::synthetic_data::*;

///Structure containing a collection of playouts which
///all have the same starting set-size and the same length in turns
///Starting set size will be denoted K, length of playout L,
pub struct PlayoutBundle {
    ///Tensor of dims NxKxM, where
    ///N is the number of example playouts, and M is the flattened matrix dimension
    pub flattened_initial_matrix_sets : Tensor,
    ///Dims NxM
    pub flattened_matrix_targets : Tensor,
    ///L-element list of tensors of dims Nx([K+i]*[K+i]) for i from 0 to L
    pub child_visit_probabilities : Vec<Tensor>,
    ///NxL index tensor for left matrix index chosen for the next move
    pub left_matrix_indices : Tensor,
    ///NxL index tensor for right matrix index chosen for next move
    pub right_matrix_indices : Tensor,
}

fn remove(named_tensor_map : &mut HashMap<String, Tensor>, key : &str) -> Result<Tensor, String> {
    named_tensor_map.remove(key).ok_or(format!("Missing key {}", key))
}

impl PlayoutBundle {
    pub fn device(&self) -> Device {
        self.left_matrix_indices.device()
    }
    pub fn get_init_set_size(&self) -> usize {
        self.flattened_initial_matrix_sets.size()[1] as usize
    }
    pub fn get_num_playouts(&self) -> usize {
        self.left_matrix_indices.size()[0] as usize
    }
    pub fn get_playout_length(&self) -> usize {
        self.left_matrix_indices.size()[1] as usize
    }
    pub fn get_final_set_size(&self) -> usize {
        self.get_init_set_size() + self.get_playout_length()
    }
    pub fn get_flattened_matrix_dim(&self) -> usize {
        self.flattened_matrix_targets.size()[1] as usize
    }
    pub fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> PlayoutBundle {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.i(batch_index_range.clone())
                                                .to_device(device).detach();

        let flattened_matrix_targets = self.flattened_matrix_targets.i(batch_index_range.clone())
                                       .to_device(device).detach();

        let child_visit_probabilities : Vec<Tensor> = self.child_visit_probabilities.iter()
                                  .map(|x|
                                        x.i(batch_index_range.clone()).to_device(device).detach())
                                  .collect();
        
        let left_matrix_indices = self.left_matrix_indices.i(batch_index_range.clone())
                                      .to_device(device).to_kind(Kind::Int64).detach();

        let right_matrix_indices = self.right_matrix_indices.i(batch_index_range.clone())
                                      .to_device(device).to_kind(Kind::Int64).detach();

        PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices
        }
    }
    fn concat_consume(a : Tensor, b : Tensor) -> Tensor {
        let result = Tensor::cat(&[a, b], 0);
        result
    }
    pub fn merge(mut self, mut other : Self) -> Self {
        let _guard = no_grad_guard();
        let flattened_initial_matrix_sets = Self::concat_consume(self.flattened_initial_matrix_sets,
                                                                  other.flattened_initial_matrix_sets);
        let flattened_matrix_targets = Self::concat_consume(
                self.flattened_matrix_targets, other.flattened_matrix_targets);

        let child_visit_probabilities =
            self.child_visit_probabilities.drain(..)
            .zip(other.child_visit_probabilities.drain(..))
                .map(|(a, b)| Self::concat_consume(a, b))
                .collect();

        let left_matrix_indices = Self::concat_consume(
                self.left_matrix_indices, other.left_matrix_indices);

        let right_matrix_indices = Self::concat_consume(
                self.right_matrix_indices, other.right_matrix_indices);

        PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices
        }
    }
    pub fn serialize(mut self, prefix : String) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();         

        result.push((format!("{}_initial_matrix_sets", prefix),
                     self.flattened_initial_matrix_sets));

        result.push((format!("{}_flattened_matrix_targets", prefix),
                    self.flattened_matrix_targets));
        
        let num_child_visit_probability_tensors = self.child_visit_probabilities.len();
        for (child_visit_probabilities, i) in self.child_visit_probabilities.drain(..)
                                              .zip(0..num_child_visit_probability_tensors) {
            let name = format!("{}_child_visit_probabilities_{}", prefix, i);
            result.push((name, child_visit_probabilities));
        }

        result.push((format!("{}_left_matrix_indices", prefix),
                    self.left_matrix_indices));
        result.push((format!("{}_right_matrix_indices", prefix),
                    self.right_matrix_indices));

        result
    }

    pub fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
           -> Result<Self, String> {
        let (initial_set_size, playout_length) = key;
        let prefix = format!("{}_{}", initial_set_size, playout_length);

        let flattened_initial_matrix_sets = remove(named_tensor_map,
                                            &format!("{}_initial_matrix_sets", prefix))?;

        let flattened_matrix_targets = remove(named_tensor_map,
                                       &format!("{}_flattened_matrix_targets", prefix))?;

        let mut child_visit_probabilities = Vec::new();
        for i in 0..playout_length {
            let name = format!("{}_child_visit_probabilities_{}", prefix, i);
            let tensor = remove(named_tensor_map, &name)?;
            child_visit_probabilities.push(tensor);
        }
        
        let left_matrix_indices = remove(named_tensor_map,
                                  &format!("{}_left_matrix_indices", prefix))?;
        
        let right_matrix_indices = remove(named_tensor_map,
                                  &format!("{}_right_matrix_indices", prefix))?;
        Result::Ok(PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices
        })
    }
}

pub struct TrainingExamples {
    ///Mapping from (init set size, playout length) to playout bundles
    pub playout_bundles : HashMap<(usize, usize), PlayoutBundle>
}

impl TrainingExamples {
    pub fn merge(&mut self, mut other : TrainingExamples) {
        for (key, other_value) in other.playout_bundles.drain() {
            if (self.playout_bundles.contains_key(&key)) {
                let my_value = self.playout_bundles.remove(&key).unwrap();
                let updated_playout_bundle = my_value.merge(other_value);
                self.playout_bundles.insert(key, updated_playout_bundle);
            } else {
                self.playout_bundles.insert(key, other_value);
            }
        }
    }
    pub fn save<T : AsRef<Path>>(mut self, path : T) -> Result<(), String> {
        let mut initial_set_sizes = Vec::new();
        let mut playout_lengths = Vec::new();
        for sizing in self.playout_bundles.keys() {
            let (initial_set_size, playout_length) = *sizing;
            initial_set_sizes.push(initial_set_size as i64);
            playout_lengths.push(playout_length as i64);
        }
        let initial_set_sizes = Tensor::try_from(initial_set_sizes.as_slice()).unwrap();
        let playout_lengths = Tensor::try_from(playout_lengths.as_slice()).unwrap();

        let mut named_tensors = Vec::new();
        named_tensors.push((String::from("initial_set_sizes"), initial_set_sizes));
        named_tensors.push((String::from("playout_lengths"), playout_lengths));

        for (sizing, playout_bundle) in self.playout_bundles.drain() {
            let (initial_set_size, playout_length) = sizing;
            let prefix = format!("{}_{}", initial_set_size, playout_length);
            let mut contained_tensors = playout_bundle.serialize(prefix);
            for elem in contained_tensors.drain(..) {
                named_tensors.push(elem);
            }
        }
        let maybe_save_result = Tensor::save_multi(&named_tensors, path);
        match (maybe_save_result) {
            Result::Ok(_) => Result::Ok(()),
            Result::Err(err) => Result::Err(format!("Could not save training examples: {}", err))
        }
    }
    pub fn load<T : AsRef<Path>>(path : T, device : Device) -> Result<TrainingExamples, String> {
        let mut named_tensors = Tensor::load_multi_with_device(path, device)
                                       .map_err(|err| format!("Error unbundling: {}", err))?;

        let mut named_tensor_map = HashMap::new(); 
        for (name, tensor) in named_tensors.drain(..) {
            named_tensor_map.insert(name, tensor.detach());
        }

        let mut result = HashMap::new();

        let initial_set_sizes = remove(&mut named_tensor_map, "initial_set_sizes")?;
        let initial_set_sizes : Vec<i64> = Vec::from(&initial_set_sizes);

        let playout_lengths = remove(&mut named_tensor_map, "playout_lengths")?;
        let playout_lengths : Vec<i64> = Vec::from(&playout_lengths);

        for i in 0..initial_set_sizes.len() {
            let initial_set_size = initial_set_sizes[i] as usize;
            let playout_length = playout_lengths[i] as usize;
            let key = (initial_set_size, playout_length);
            let playout_bundle = PlayoutBundle::load(&mut named_tensor_map, key)?;
            result.insert(key, playout_bundle);
        }
        Result::Ok(TrainingExamples {
            playout_bundles : result
        })
    }
}

pub struct TrainingExamplesBuilder {
    ///Mapping from (init set size, playout length) to playout bundles
    pub playout_bundles : HashMap<(usize, usize), PlayoutBundleBuilder>,
    pub m : usize
}

pub struct PlayoutBundleBuilder {
    pub playouts : Vec<PlayoutBuilder>,
}


impl PlayoutBundleBuilder {
    pub fn new() -> Self {
        PlayoutBundleBuilder {
            playouts : Vec::new(),
        }
    }
    fn construct_tensor<T : Element>(value : Vec<T>, dims : &[i64]) -> Tensor {
        let unshaped = Tensor::try_from(value).unwrap();
        let shaped = unshaped.reshape(dims);
        shaped
    }
    pub fn add_annotated_game_path(&mut self, annotated_game_path : AnnotatedGamePath) {
        let playout_bundle = PlayoutBuilder::from_annotated_game_path(annotated_game_path);
        self.playouts.push(playout_bundle);
    }
    pub fn build(mut self, k : usize, l : usize, m : usize) -> PlayoutBundle {
        let n = self.playouts.len();

        //Capacity reservation
        let mut flattened_initial_matrix_sets = Vec::with_capacity(n * k * m);
        let mut flattened_matrix_targets = Vec::with_capacity(n * m);
        let mut child_visit_probabilities = Vec::new();
        for i in 0..l {
            let dim = k + i;
            let child_visit_probabilities_entry = Vec::with_capacity(n * dim * dim);
            child_visit_probabilities.push(child_visit_probabilities_entry);
        }
        let mut left_matrix_indices = Vec::with_capacity(n * l);
        let mut right_matrix_indices = Vec::with_capacity(n * l);

        //Appending
        for mut playout in self.playouts.drain(..) {
            flattened_initial_matrix_sets.append(&mut playout.flattened_initial_matrix_set);
            flattened_matrix_targets.append(&mut playout.flattened_matrix_target);
            for i in 0..l {
                child_visit_probabilities[i].append(&mut playout.child_visit_probabilities[i]);
            }
            left_matrix_indices.append(&mut playout.left_matrix_indices);
            right_matrix_indices.append(&mut playout.right_matrix_indices);
        }

        let n = n as i64;
        let k = k as i64;
        let l = l as i64;
        let m = m as i64;

        //Creating tensors
        let flattened_initial_matrix_sets = Self::construct_tensor(flattened_initial_matrix_sets, &[n, k, m]);
        let flattened_matrix_targets = Self::construct_tensor(flattened_matrix_targets, &[n, m]);

        let mut child_visit_probabilities_tensored = Vec::new();
        for (child_visit_array, i) in child_visit_probabilities.drain(..).zip(0..l) {
            let dim = k + i;
            let child_visit_probabilities_entry = Self::construct_tensor(child_visit_array, &[n, dim, dim]);
            child_visit_probabilities_tensored.push(child_visit_probabilities_entry);
        }
        let child_visit_probabilities = child_visit_probabilities_tensored; 
        
        let left_matrix_indices = Self::construct_tensor(left_matrix_indices, &[n, l]);
        let right_matrix_indices = Self::construct_tensor(right_matrix_indices, &[n, l]);

        PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices
        }
    }
    pub fn shuffle<R : Rng + ?Sized>(&mut self, rng : &mut R) {
        self.playouts.shuffle(rng);
    }
}

pub struct PlayoutBuilder {
    ///Dims: K x M
    pub flattened_initial_matrix_set : Vec<f32>,
    ///Dims: M
    pub flattened_matrix_target : Vec<f32>,
    ///Dims: L x (K+i) x (K+i) for i in [0, L)
    pub child_visit_probabilities : Vec<Vec<f32>>,
    ///Dims: L
    pub left_matrix_indices : Vec<u8>,
    ///Dims: L
    pub right_matrix_indices : Vec<u8>
}

impl PlayoutBuilder {
    pub fn from_annotated_game_path(mut annotated_game_path : AnnotatedGamePath) -> Self {
        let flattened_initial_matrix_set = annotated_game_path.matrix_set.to_flattened_vec();
        let flattened_matrix_target = annotated_game_path.target_matrix.into_raw_vec();
        
        let mut left_matrix_indices = Vec::new();
        let mut right_matrix_indices = Vec::new();
        let mut child_visit_probabilities = Vec::new();
        for node in annotated_game_path.nodes.drain(..) {
            left_matrix_indices.push(node.left_index as u8);
            right_matrix_indices.push(node.right_index as u8);
            child_visit_probabilities.push(node.child_visit_probabilities.into_raw_vec());
        }

        PlayoutBuilder {
            flattened_initial_matrix_set,
            flattened_matrix_target,
            left_matrix_indices,
            right_matrix_indices,
            child_visit_probabilities
        }
    }
}

impl TrainingExamplesBuilder {
    pub fn new(params : &Params) -> TrainingExamplesBuilder {
        let playout_bundles = HashMap::new();
        let m = params.get_flattened_matrix_dim() as usize;
        TrainingExamplesBuilder {
            playout_bundles,
            m
        }
    }
    pub fn add_annotated_game_path(&mut self, annotated_game_path : AnnotatedGamePath) {
        let init_set_size = annotated_game_path.matrix_set.len();
        let playout_length = annotated_game_path.nodes.len();
        let key = (init_set_size, playout_length);
        if !self.playout_bundles.contains_key(&key) {
            self.playout_bundles.insert(key, PlayoutBundleBuilder::new());
        }
        self.playout_bundles.get_mut(&key).unwrap().add_annotated_game_path(annotated_game_path);
    }
    pub fn build(mut self) -> TrainingExamples {
        let mut playout_bundles = HashMap::new();
        for (size_key, playout_bundle_builder) in self.playout_bundles.drain() {
            let (k, l) = size_key;
            let playout_bundle = playout_bundle_builder.build(k, l, self.m);
            playout_bundles.insert(size_key, playout_bundle);
        }
        TrainingExamples {
            playout_bundles
        }
    }
}
