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
use crate::training_examples::*;

///Structure containing a collection of playout sketches
///all of which have the same starting set-size (K),
///and the same length in turns (L). Unlike
///PlayoutBundle, no matrices are associated with these,
///they're just "sketches" of the shapes of certain games
pub struct PlayoutSketchBundle {
    ///L-element list of tensors of dims Nx[K+i]x[K+i] for i from 0 to L
    pub child_visit_probabilities : Vec<Tensor>,
    ///NxL index tensor for left matrix index chosen for the next move
    pub left_matrix_indices : Tensor,
    ///NxL index tensor for right matrix index chosen for next move
    pub right_matrix_indices : Tensor,
}

fn remove(named_tensor_map : &mut HashMap<String, Tensor>, key : &str) -> Result<Tensor, String> {
    named_tensor_map.remove(key).ok_or(format!("Missing key {}", key))
}

impl PlayoutSketchBundle {
    pub fn to_device(&self, device : Device) -> Self {
        let child_visit_probabilities = self.child_visit_probabilities.iter()
                                            .map(move |x| x.to_device(device))
                                            .collect();
        let left_matrix_indices = self.left_matrix_indices.to_device(device);
        let right_matrix_indices = self.right_matrix_indices.to_device(device);
        Self {
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices,
        }
    }
    pub fn from_single_annotated_game_path(annotated_game_path : AnnotatedGamePath) -> Self {
        let mut playout_sketch_bundle_builder = PlayoutSketchBundleBuilder::new(annotated_game_path.get_num_turns());
        playout_sketch_bundle_builder.add_annotated_game_path(annotated_game_path);
        playout_sketch_bundle_builder.build()
    }
    pub fn shallow_clone(&self) -> Self {
        let child_visit_probabilities = self.child_visit_probabilities.iter()
                                        .map(|x| x.shallow_clone())
                                        .collect();
        let left_matrix_indices = self.left_matrix_indices.shallow_clone();
        let right_matrix_indices = self.right_matrix_indices.shallow_clone();
        Self {
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices,
        }
    }
    pub fn device(&self) -> Device {
        self.left_matrix_indices.device()
    }
    pub fn get_init_set_size(&self) -> usize {
        let k = self.child_visit_probabilities[0].size()[1];
        k as usize
    }
    pub fn get_playout_length(&self) -> usize {
        self.left_matrix_indices.size()[1] as usize
    }
    fn concat_consume(a : Tensor, b : Tensor) -> Tensor {
        let result = Tensor::cat(&[a, b], 0);
        result
    }
}
impl PlayoutBundleLike for PlayoutSketchBundle {
    fn get_num_playouts(&self) -> usize {
        self.left_matrix_indices.size()[0] as usize
    }
    fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> PlayoutSketchBundle {
        let child_visit_probabilities : Vec<Tensor> = self.child_visit_probabilities.iter()
                              .map(|x|
                                    x.i(batch_index_range.clone()).to_device(device).detach())
                              .collect();
        
        let left_matrix_indices = self.left_matrix_indices.i(batch_index_range.clone())
                                      .to_device(device).to_kind(Kind::Int64).detach();

        let right_matrix_indices = self.right_matrix_indices.i(batch_index_range.clone())
                                      .to_device(device).to_kind(Kind::Int64).detach();

        PlayoutSketchBundle {
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices
        }
    }
    fn merge(mut self, mut other : Self) -> Self {
        let _guard = no_grad_guard();
    
        let child_visit_probabilities =
            self.child_visit_probabilities.drain(..)
            .zip(other.child_visit_probabilities.drain(..))
                .map(|(a, b)| Self::concat_consume(a, b))
                .collect();

        let left_matrix_indices = Self::concat_consume(
                self.left_matrix_indices, other.left_matrix_indices);

        let right_matrix_indices = Self::concat_consume(
                self.right_matrix_indices, other.right_matrix_indices);

        PlayoutSketchBundle {
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices,
        }
    }
    fn serialize(mut self, prefix : String) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();

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
    fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
        -> Result<Self, String> {
        let (initial_set_size, playout_length) = key;
        let prefix = format!("{}_{}", initial_set_size, playout_length);
    
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
        Ok(Self {
            child_visit_probabilities,
            left_matrix_indices,
            right_matrix_indices,
        })
    }
}

pub struct SketchExamplesBuilder {
    ///Mapping from (init set size, playout length) to playout bundle builders
    pub playout_bundles : HashMap<(usize, usize), PlayoutSketchBundleBuilder>,
}

impl SketchExamplesBuilder {
    pub fn new() -> Self {
        Self {
            playout_bundles : HashMap::new(),
        }
    }
    pub fn add_annotated_game_path(&mut self, annotated_game_path : AnnotatedGamePath) {
        let K = annotated_game_path.get_initial_set_size();
        let L = annotated_game_path.get_num_turns();
        let key = (K, L); 
        if !self.playout_bundles.contains_key(&key) {
            self.playout_bundles.insert(key, PlayoutSketchBundleBuilder::new(L));
        }
        self.playout_bundles.get_mut(&key).unwrap().add_annotated_game_path(annotated_game_path);
    }
    pub fn build(mut self) -> TrainingExamples<PlayoutSketchBundle> {
        let mut playout_bundles = HashMap::new();
        for (key, bundle_builder) in self.playout_bundles.drain() {
            playout_bundles.insert(key, bundle_builder.build());
        }
        TrainingExamples {
            playout_bundles,
        }
    }
}

pub struct PlayoutSketchBundleBuilder {
    ///L-element list of tensors of dims Nx([K+i]*[K+i]) for i from 0 to L
    pub child_visit_probabilities : Vec<Vec<f32>>,
    //Both NxL when reshaped, but they're flat.
    pub left_matrix_indices : Vec<u8>,
    pub right_matrix_indices : Vec<u8>,
    pub num_examples : usize,
}

impl PlayoutSketchBundleBuilder {
    pub fn new(L : usize) -> Self {
        let mut child_visit_probabilities = Vec::new();
        for _ in 0..L {
            child_visit_probabilities.push(Vec::new());
        }
        Self {
            child_visit_probabilities,
            left_matrix_indices : Vec::new(),
            right_matrix_indices : Vec::new(),
            num_examples : 0,
        }
    }
    pub fn add_annotated_game_path(&mut self, mut annotated_game_path : AnnotatedGamePath) {
        for (i, node) in annotated_game_path.nodes.drain(..).enumerate() {
            self.left_matrix_indices.push(node.left_index as u8);
            self.right_matrix_indices.push(node.right_index as u8);

            let mut child_visit_probabilities_to_add = node.child_visit_probabilities.into_raw_vec();
            self.child_visit_probabilities[i].append(&mut child_visit_probabilities_to_add);
        }
        self.num_examples += 1;
    }
    pub fn build(mut self) -> PlayoutSketchBundle {
        let n = self.num_examples as i64;
        let n_times_l = self.left_matrix_indices.len() as i64;
        let l = n_times_l / n;

        let left_matrix_indices = Tensor::try_from(self.left_matrix_indices).unwrap()
                                         .reshape(&[n, l]);
        let right_matrix_indices = Tensor::try_from(self.right_matrix_indices).unwrap()
                                         .reshape(&[n, l]);

        let mut child_visit_probabilities = Vec::new();
        for child_visit_probability in self.child_visit_probabilities {
            let k_plus_i_squared = (child_visit_probability.len() as i64) / n;
            let k_plus_i = (k_plus_i_squared as f64).sqrt() as i64;
            let child_visit_probability = Tensor::try_from(child_visit_probability).unwrap()
                                                .reshape(&[n, k_plus_i, k_plus_i]);
            child_visit_probabilities.push(child_visit_probability);
        }
        PlayoutSketchBundle {
            left_matrix_indices,
            right_matrix_indices,
            child_visit_probabilities,
        }
    }
}
