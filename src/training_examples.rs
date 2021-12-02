use tch::{nn, nn::Init, nn::Module, Tensor, nn::Sequential};
use crate::game_data::*;
use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;
use ndarray::*;
use crate::array_utils::*;
use std::convert::{TryFrom, TryInto};
use std::path::Path;
use crate::params::*;

pub struct TrainingExamples {
    ///Mapping from set size to a K-element list of tensors of dims NxM,
    ///where K is the set size, N is the number of examples,
    ///and M is the flattened matrix dimension
    pub flattened_matrix_sets : HashMap<usize, Vec<Tensor>>,
    ///Mapping from set size to a tensor of dims NxM
    pub flattened_matrix_targets : HashMap<usize, Tensor>,
    ///Mapping from set size to a tensor of dims Nx(K*K)
    pub child_visit_probabilities : HashMap<usize, Tensor>,
}

struct TrainingExamplesBuilder {
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

impl TrainingExamplesBuilder {
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


    fn permute<R : Rng + ?Sized>(mut flattened_matrix_set : Vec<Array1<f32>>, 
                                 child_visit_probabilities : Array2<f32>,
                                 rng : &mut R) -> (Vec<Array1<f32>>, Array2<f32>) {
        let k = flattened_matrix_set.len();
        let mut result_flattened_matrix_set = Vec::new();
        for _ in 0..k {
            result_flattened_matrix_set.push(Option::None);
        }
        let mut result_visit_probabilities = child_visit_probabilities.clone();

        let mut permutation : Vec<usize> = (0..k).collect();
        permutation.shuffle(rng);

        for (flattened_matrix, i) in flattened_matrix_set.drain(..).zip(0..k) {
            let dest_index = permutation[i];
            result_flattened_matrix_set[dest_index] = Option::Some(flattened_matrix);
        }

        let result_flattened_matrix_set = result_flattened_matrix_set.drain(..).map(|x| x.unwrap()).collect();

        for i in 0..k {
            let dest_i = permutation[i];
            for j in 0..k {
                let dest_j = permutation[j];
                result_visit_probabilities[[dest_i, dest_j]] = child_visit_probabilities[[i, j]];
            }
        }

        (result_flattened_matrix_set, result_visit_probabilities)
    }

    fn add_game_data<R : Rng + ?Sized>(&mut self, mut game_data : GameData, rng : &mut R) {
        let flattened_matrix_target = game_data.flattened_matrix_target;
        for (flattened_matrix_set, child_visit_probabilities) in 
            game_data.flattened_matrix_sets.drain(..)
        .zip(game_data.child_visit_probabilities.drain(..)) {

            let (flattened_matrix_set, child_visit_probabilities) = 
                Self::permute(flattened_matrix_set, child_visit_probabilities, rng);

            self.add_matrix_data(flattened_matrix_set, flattened_matrix_target.view(), 
                                 child_visit_probabilities);
        }
    }

    fn add_matrix_data(&mut self, mut flattened_matrix_set : Vec<Array1<f32>>, 
                       flattened_matrix_target : ArrayView1<f32>, child_visit_probabilities : Array2<f32>) {

        let set_size = flattened_matrix_set.len();
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
        for (flattened_matrix, i) in flattened_matrix_set.drain(..).zip(0..set_size) {
            matrix_set_dest[i].extend_from_slice(flattened_matrix.as_slice().unwrap());
        }

        self.flattened_matrix_targets.get_mut(&set_size).unwrap()
            .extend_from_slice(flattened_matrix_target.as_slice().unwrap());

        self.child_visit_probabilities.get_mut(&set_size).unwrap()
            .extend_from_slice(child_visit_probabilities.as_slice().unwrap());

        let prev_num_samples = self.num_samples.remove(&set_size).unwrap(); 
        self.num_samples.insert(set_size, prev_num_samples + 1);
    }

}

impl TrainingExamples {
    pub fn from_game_data<R : Rng + ?Sized>(params : &Params, mut game_datas : Vec<GameData>, rng : &mut R) -> TrainingExamples {
        let flattened_matrix_sets = HashMap::new();
        let flattened_matrix_targets = HashMap::new();
        let child_visit_probabilities = HashMap::new();
        let num_samples = HashMap::new();
        let mut builder = TrainingExamplesBuilder {
            flattened_matrix_sets,
            flattened_matrix_targets,
            child_visit_probabilities,
            num_samples,
            m : params.get_flattened_matrix_dim() as usize
        };

        for game_data in game_datas.drain(..) {
            builder.add_game_data(game_data, rng);
        }

        builder.build(rng)
    }

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
    pub fn load<T : AsRef<Path>>(path : T) -> Result<TrainingExamples, String> {
        let maybe_named_tensors = Tensor::load_multi(path);
        match (maybe_named_tensors) {
            Result::Ok(mut named_tensors) => {
                let mut flattened_matrix_sets = HashMap::new();
                let mut flattened_matrix_targets = HashMap::new();
                let mut child_visit_probabilities = HashMap::new();

                let mut named_tensor_map = HashMap::new(); 
                for (name, tensor) in named_tensors.drain(..) {
                    named_tensor_map.insert(name, tensor);
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
