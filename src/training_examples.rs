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
use crate::rollout_states::*;
use crate::synthetic_data::*;
use crate::playout_sketches::*;

///Structure containing a collection of playouts which
///all have the same starting set-size and the same length in turns
///Starting set size will be denoted K, length of playout L,
pub struct PlayoutBundle {
    ///Tensor of dims NxKxM, where
    ///N is the number of example playouts, and M is the flattened matrix dimension
    pub flattened_initial_matrix_sets : Tensor,
    ///Dims NxM
    pub flattened_matrix_targets : Tensor,
    ///The sketches that we've specified starting+ending matrices for
    pub sketch_bundle : PlayoutSketchBundle,
}

fn remove(named_tensor_map : &mut HashMap<String, Tensor>, key : &str) -> Result<Tensor, String> {
    named_tensor_map.remove(key).ok_or(format!("Missing key {}", key))
}

impl PlayoutBundle {
    pub fn to_device(&self, device : Device) -> Self {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.to_device(device);
        let flattened_matrix_targets = self.flattened_matrix_targets.to_device(device);
        let sketch_bundle = self.sketch_bundle.to_device(device);
        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            sketch_bundle,
        }
    }
    pub fn standardize(&self) -> PlayoutBundle {
        let n = self.get_num_playouts() as i64;
        let k = self.get_init_set_size() as i64;
        let m = self.flattened_matrix_targets.size()[1];
        let m_sqrt = (m as f64).sqrt() as i64;
        let reshaped_matrix_targets = self.flattened_matrix_targets.reshape(&[n, m_sqrt, m_sqrt]);

        //N x M_sqrt x M_sqrt
        let Q = derive_orthonormal_basis_changes_from_target_matrices(&reshaped_matrix_targets);
        let Q_T = Q.transpose(1, 2); 

        let transformed_matrix_targets = Q_T.matmul(&reshaped_matrix_targets).matmul(&Q);

        let Q = Q.reshape(&[n, 1, m_sqrt, m_sqrt]);
        let Q_T = Q_T.reshape(&[n, 1, m_sqrt, m_sqrt]);

        let reshaped_initial_matrix_sets = self.flattened_initial_matrix_sets.reshape(&[n, k, m_sqrt, m_sqrt]);
        
        let transformed_initial_matrix_sets = Q_T.matmul(&reshaped_initial_matrix_sets).matmul(&Q);

        let flattened_initial_matrix_sets = transformed_initial_matrix_sets.reshape(&[n, k, m]);
        let flattened_matrix_targets = transformed_matrix_targets.reshape(&[n, m]);
        
        let sketch_bundle = self.sketch_bundle.shallow_clone();

        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            sketch_bundle,
        }
    }
    ///Lifts a PlayoutSketchBundle to a PlayoutBundle by selecting random
    ///starting matrices acording to the passed parameters
    pub fn from_sketch_bundle(params : &Params, sketch_bundle : PlayoutSketchBundle) -> Self {
        let num_playouts = sketch_bundle.get_num_playouts() as i64;
        let init_set_size = sketch_bundle.get_init_set_size() as i64;
        let flattened_matrix_dim = params.get_flattened_matrix_dim();

        //Generate N * K random matrices
        let num_random_matrices = num_playouts * init_set_size;
        //Dims (N * K) x sqrt(M) x sqrt(M)
        let random_matrices = params.generate_random_matrices(num_random_matrices as usize);
        let flattened_initial_matrix_sets = random_matrices.reshape(&[num_playouts, init_set_size, flattened_matrix_dim]);

        
        //Generate some matrices of zeroes for an initial fake 'target'
        //TODO: We could remove the need for extra computations here by adding an
        //extra layer of abstraction around rollouts with targets vs rollouts without
        let targets_dim = [num_playouts, flattened_matrix_dim];
        let fake_targets = Tensor::zeros(&targets_dim, (Kind::Float, sketch_bundle.device()));

        //We'll fix up the playout bundle's target, no worries
        let mut result = PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets : fake_targets,
            sketch_bundle,
        };
        let mut target_finding_rollout = RolloutStates::from_playout_bundle_initial_state(&result);

        let mut left_matrix_indices = result.sketch_bundle.left_matrix_indices.unbind(1);
        left_matrix_indices.reverse();

        let mut right_matrix_indices = result.sketch_bundle.right_matrix_indices.unbind(1);
        right_matrix_indices.reverse();
        //Roll forward the target-finding rollout 
        let playout_length = result.get_playout_length();
        for i in 0..(playout_length - 1) {
            let left_indices = left_matrix_indices.pop().unwrap();
            let right_indices = right_matrix_indices.pop().unwrap();
            target_finding_rollout = target_finding_rollout.manual_step(&left_indices, &right_indices);
        }
        //Perform the final step, for which we'll only need the diff
        let left_indices = left_matrix_indices.pop().unwrap();
        let right_indices = right_matrix_indices.pop().unwrap();
        let final_step_diff = target_finding_rollout.perform_moves_diff(&left_indices, &right_indices);

        result.flattened_matrix_targets = final_step_diff.matrices.reshape(&targets_dim);

        result
    }
    pub fn device(&self) -> Device {
        self.sketch_bundle.device()
    }
    pub fn get_init_set_size(&self) -> usize {
        self.flattened_initial_matrix_sets.size()[1] as usize
    }
    pub fn get_playout_length(&self) -> usize {
        self.sketch_bundle.get_playout_length()
    }
    pub fn get_final_set_size(&self) -> usize {
        self.get_init_set_size() + self.get_playout_length()
    }
    pub fn get_flattened_matrix_dim(&self) -> usize {
        self.flattened_matrix_targets.size()[1] as usize
    }
    fn concat_consume(a : Tensor, b : Tensor) -> Tensor {
        let result = Tensor::cat(&[a, b], 0);
        result
    }
}
impl PlayoutBundleLike for PlayoutBundle {
    fn get_num_playouts(&self) -> usize {
        self.sketch_bundle.get_num_playouts()
    }
    fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> PlayoutBundle {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.i(batch_index_range.clone())
                                                .to_device(device).detach();

        let flattened_matrix_targets = self.flattened_matrix_targets.i(batch_index_range.clone())
                                       .to_device(device).detach();

        let sketch_bundle = self.sketch_bundle.grab_batch(batch_index_range, device);

        PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            sketch_bundle,
        }
    }
    fn merge(mut self, mut other : Self) -> Self {
        let _guard = no_grad_guard();
        let flattened_initial_matrix_sets = Self::concat_consume(self.flattened_initial_matrix_sets,
                                                                  other.flattened_initial_matrix_sets);
        let flattened_matrix_targets = Self::concat_consume(
                self.flattened_matrix_targets, other.flattened_matrix_targets);

        let sketch_bundle = self.sketch_bundle.merge(other.sketch_bundle);

        PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            sketch_bundle,
        }
    }
    fn serialize(mut self, prefix : String) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();         

        let mut sketch_entries = self.sketch_bundle.serialize(prefix.clone());

        result.push((format!("{}_initial_matrix_sets", prefix),
                     self.flattened_initial_matrix_sets));

        result.push((format!("{}_flattened_matrix_targets", prefix),
                    self.flattened_matrix_targets));

        result.append(&mut sketch_entries);

        result
    }

    fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
           -> Result<Self, String> {
        let (initial_set_size, playout_length) = key;
        let prefix = format!("{}_{}", initial_set_size, playout_length);

        let flattened_initial_matrix_sets = remove(named_tensor_map,
                                            &format!("{}_initial_matrix_sets", prefix))?;

        let flattened_matrix_targets = remove(named_tensor_map,
                                       &format!("{}_flattened_matrix_targets", prefix))?;
        
        let sketch_bundle = PlayoutSketchBundle::load(named_tensor_map, key)?;

        Result::Ok(PlayoutBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
            sketch_bundle,
        })
    }
}

pub trait PlayoutBundleLike : Sized {
    fn get_num_playouts(&self) -> usize;
    fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> Self;
    fn merge(self, other : Self) -> Self;
    fn serialize(self, prefix : String) -> Vec<(String, Tensor)>;
    fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
        -> Result<Self, String>;
}

pub struct TrainingExamples<BundleType : PlayoutBundleLike> {
    ///Mapping from (init set size, playout length) to playout bundles
    pub playout_bundles : HashMap<(usize, usize), BundleType>,
}

impl <BundleType : PlayoutBundleLike> TrainingExamples<BundleType> {
    pub fn merge(&mut self, mut other : TrainingExamples<BundleType>) {
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
    pub fn load<T : AsRef<Path>>(path : T, device : Device) -> Result<Self, String> {
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
            let playout_bundle = BundleType::load(&mut named_tensor_map, key)?;
            result.insert(key, playout_bundle);
        }
        Result::Ok(TrainingExamples {
            playout_bundles : result
        })
    }
}
