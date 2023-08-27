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
use crate::matrix_sets::*;
use crate::network_config::*;
use crate::rollout_states::*;
use crate::synthetic_data::*;
use crate::playout_sketches::*;
use crate::matrix_bundle::*;

///Structure containing a collection of playouts which
///all have the same starting set-size and the same length in turns
///Starting set size will be denoted K, length of playout L,
pub struct PlayoutBundle {
    ///The starting + ending matrices for the playout bundle
    pub matrix_bundle : MatrixBundle,
    ///The sketches that we've specified starting+ending matrices for
    pub sketch_bundle : PlayoutSketchBundle,
}

fn remove(named_tensor_map : &mut HashMap<String, Tensor>, key : &str) -> Result<Tensor, String> {
    named_tensor_map.remove(key).ok_or(format!("Missing key {}", key))
}

impl PlayoutBundle {
    pub fn to_device(&self, device : Device) -> Self {
        let matrix_bundle = self.matrix_bundle.to_device(device);
        let sketch_bundle = self.sketch_bundle.to_device(device);
        Self {
            matrix_bundle,
            sketch_bundle,
        }
    }
    pub fn standardize(&self) -> PlayoutBundle {
        let matrix_bundle = self.matrix_bundle.standardize();
        let sketch_bundle = self.sketch_bundle.shallow_clone();

        Self {
            matrix_bundle,
            sketch_bundle,
        }
    }
    ///Lifts a PlayoutSketchBundle to a PlayoutBundle using the given
    ///collection of random initial matrices (dims (N*K) x sqrt(M) x sqrt(M))
    pub fn from_initial_matrices_and_sketch_bundle(initial_matrices : Tensor, sketch_bundle : PlayoutSketchBundle) -> Self {
        let num_playouts = sketch_bundle.get_num_playouts() as i64;
        let init_set_size = sketch_bundle.get_init_set_size() as i64;
        let matrix_dim = initial_matrices.size()[1];

        let initial_matrices = initial_matrices.reshape(&[num_playouts, init_set_size, matrix_dim, matrix_dim]);
        let initial_matrices = MatrixSets::new(initial_matrices);

        let flattened_matrix_targets = initial_matrices.get_flattened_targets_from_moves(
                                    &sketch_bundle.left_matrix_indices,
                                    &sketch_bundle.right_matrix_indices,
                                );
        let flattened_initial_matrix_sets = initial_matrices.get_flattened_matrices();

        let matrix_bundle = MatrixBundle {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        };

        PlayoutBundle {
            matrix_bundle,
            sketch_bundle,
        }
    }

    ///Lifts a PlayoutSketchBundle to a PlayoutBundle by selecting random
    ///starting matrices acording to the passed parameters
    pub fn from_sketch_bundle(params : &Params, sketch_bundle : PlayoutSketchBundle) -> Self {
        let num_playouts = sketch_bundle.get_num_playouts() as i64;
        let init_set_size = sketch_bundle.get_init_set_size() as i64;

        //Generate N * K random matrices
        let num_random_matrices = num_playouts * init_set_size;
        //Dims (N * K) x sqrt(M) x sqrt(M)
        let random_matrices = params.generate_random_matrices(num_random_matrices as usize);
        Self::from_initial_matrices_and_sketch_bundle(random_matrices, sketch_bundle)
    }

    pub fn device(&self) -> Device {
        self.sketch_bundle.device()
    }
    pub fn get_init_set_size(&self) -> usize {
        self.matrix_bundle.get_init_set_size()
    }
    pub fn get_playout_length(&self) -> usize {
        self.sketch_bundle.get_playout_length()
    }
    pub fn get_final_set_size(&self) -> usize {
        self.get_init_set_size() + self.get_playout_length()
    }
    pub fn get_flattened_matrix_dim(&self) -> usize {
        self.matrix_bundle.get_flattened_matrix_dim()
    }
}
impl PlayoutBundleLike for PlayoutBundle {
    fn get_num_playouts(&self) -> usize {
        self.sketch_bundle.get_num_playouts()
    }
    fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> PlayoutBundle {
        let matrix_bundle = self.matrix_bundle.grab_batch(batch_index_range.clone(), device);
        let sketch_bundle = self.sketch_bundle.grab_batch(batch_index_range, device);

        PlayoutBundle {
            matrix_bundle,
            sketch_bundle,
        }
    }
    fn merge(mut self, mut other : Self) -> Self {
        let matrix_bundle = self.matrix_bundle.merge(other.matrix_bundle);
        let sketch_bundle = self.sketch_bundle.merge(other.sketch_bundle);

        PlayoutBundle {
            matrix_bundle,
            sketch_bundle,
        }
    }
    fn serialize(mut self, prefix : String) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();         

        let mut matrix_entries = self.matrix_bundle.serialize(prefix.clone());
        let mut sketch_entries = self.sketch_bundle.serialize(prefix);

        result.append(&mut matrix_entries);
        result.append(&mut sketch_entries);

        result
    }

    fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
           -> Result<Self, String> {
        let matrix_bundle = MatrixBundle::load(named_tensor_map, key)?;
        let sketch_bundle = PlayoutSketchBundle::load(named_tensor_map, key)?;

        Result::Ok(PlayoutBundle {
            matrix_bundle,
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
