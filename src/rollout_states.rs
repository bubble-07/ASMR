use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use crate::network::*;
use crate::neural_utils::*;
use crate::array_utils::*;
use crate::params::*;
use crate::training_examples::*;
use crate::network_config::*;
use ndarray::*;
use std::convert::TryFrom;
use std::fmt;

use rand::Rng;
use rand::seq::SliceRandom;
use core::iter::Sum;
use std::iter::zip;


///Snapshot state of R rollouts for each matrix product among an initial set of k matrices,
///where each snapshot takes place at some number of turns after the initial turn
pub struct RolloutStates {
    ///1D tensor of size R, containing minimal distances along each rollout path
    ///at the most recent move in each rollout
    pub min_distances : Tensor,

    ///Flattened target matrices (dims R x m * m)
    pub flattened_targets : Tensor,

    ///The matrices in each set, each of dims MxM, R x (k+t) of 'em
    ///R x (k + t) x M x M
    pub matrices : Tensor,

    ///The remaining number of turns
    pub remaining_turns : usize,
}

impl fmt::Display for RolloutStates {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "set: {} \n target: {} \n turns: {}", 
               self.matrices.to_string(80).unwrap(), 
               self.flattened_targets.to_string(80).unwrap(), 
               self.remaining_turns)
    }
}

pub struct RolloutStatesDiff {
    ///1D tensor of size R, containing minimal distances along each rollout path
    ///at the most recent move in each rollout.
    ///Straight up replaces the rollout state min distances.
    pub min_distances : Tensor,
    ///The matrices which were added at this step
    ///R x 1 x M x M
    pub matrices : Tensor,
}

impl fmt::Display for RolloutStatesDiff {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "added: {}", self.matrices.to_string(80).unwrap())
    }
}

impl RolloutStatesDiff {
    pub fn split_to_singles(self) -> Vec<RolloutStatesDiff> {
        let min_distances = self.min_distances.split(1, 0);
        let matrices = self.matrices.split(1, 0);
        zip(min_distances, matrices)
        .map(|(min_distances, matrices)| {
            RolloutStatesDiff {
                min_distances,
                matrices
            }
        }).collect()
    }
    pub fn get_flattened_added_matrices(&self) -> Tensor {
        let r = self.matrices.size()[0];
        let m = self.matrices.size()[2];
        self.matrices.reshape(&[r, m * m])
    }
}

impl RolloutStates {
    pub fn shallow_clone(&self) -> Self {
        Self {
            min_distances : self.min_distances.shallow_clone(),
            flattened_targets : self.flattened_targets.shallow_clone(),
            matrices : self.matrices.shallow_clone(),
            remaining_turns : self.remaining_turns,
        }
    }
    pub fn complete_random_rollouts(self) -> Self {
        let R = self.get_num_rollouts();
        let mut left_indices = Tensor::zeros(&[R], (Kind::Int64, self.matrices.device()));
        let mut right_indices = left_indices.zeros_like();

        let mut result = self;
        while result.remaining_turns > 0 {
            let num_matrices = result.get_num_matrices();
            //Generate random moves for the R rollouts
            left_indices.random_to_(num_matrices);
            right_indices.random_to_(num_matrices);
            
            result = result.manual_step(&left_indices, &right_indices);
        }
        result
    }
    pub fn apply_diff(self, diff : &RolloutStatesDiff) -> Self {
        let remaining_turns = self.remaining_turns - 1;
        let matrices = Tensor::concat(&[self.matrices, diff.matrices.shallow_clone()], 1);
        RolloutStates {
            min_distances : diff.min_distances.shallow_clone(),
            flattened_targets : self.flattened_targets,
            remaining_turns,
            matrices,
        }
    }

    pub fn perform_all_moves(self) -> Self {
        let device = self.matrices.device();
        let num_matrices = self.get_num_matrices();
        let num_children = num_matrices * num_matrices;

        let (left_indices, right_indices) = generate_2d_index_tensor_span(num_matrices, device);

        //Expand and perform moves
        let result = self.expand(num_children as usize);
        let result = result.manual_step(&left_indices, &right_indices);
        result
    }

    ///Merges a collection of rollout states with the same values of
    ///k+t and m, but possibly with differing numbers of rollouts
    ///and/or differing numbers of remaining turns (the output
    ///number of remaining turns will be the maximum of all inputs).
    pub fn merge(states : Vec<RolloutStates>) -> RolloutStates {
        let min_distances : Vec<Tensor> = states.iter().map(|x| x.min_distances.shallow_clone()).collect();
        let flattened_targets : Vec<Tensor> = states.iter().map(|x| x.flattened_targets.shallow_clone()).collect();
        let matrices : Vec<Tensor> = states.iter().map(|x| x.matrices.shallow_clone()).collect();

        let min_distances = Tensor::concat(&min_distances, 0);
        let flattened_targets = Tensor::concat(&flattened_targets, 0);
        let matrices = Tensor::concat(&matrices, 0);
        
        let remaining_turns = states.iter().map(|x| x.remaining_turns).max().unwrap();

        RolloutStates {
            min_distances,
            flattened_targets,
            matrices,
            remaining_turns,
        }
    }

    ///Splits to a collection of rollout states where each
    ///rollout state object contains the same number of rollouts
    pub fn split(self, split_sizes : &[i64]) -> Vec<RolloutStates> {
        let min_distances = self.min_distances.split_with_sizes(split_sizes, 0);
        let flattened_targets = self.flattened_targets.split_with_sizes(split_sizes, 0);
        let matrices = self.matrices.split_with_sizes(split_sizes, 0);
        let remaining_turns = self.remaining_turns;

        zip(zip(min_distances, flattened_targets), matrices)
        .map(|((min_distances, flattened_targets), matrices)| 
             RolloutStates {
                 min_distances,
                 flattened_targets,
                 matrices,
                 remaining_turns,
             }
            )
        .collect()
    }

    pub fn get_num_rollouts(&self) -> i64 {
        self.matrices.size()[0]
    }
    pub fn get_matrix_size(&self) -> i64 {
        self.matrices.size()[2]
    }
    pub fn get_num_matrices(&self) -> i64 {
        self.matrices.size()[1]
    }

    fn expand_single_indices(&self, left_index : usize, right_index : usize) -> (Tensor, Tensor) {
        let device = self.matrices.device();
        let left_indices = Tensor::try_from(&vec![left_index as i64]).unwrap().to_device(device);
        let right_indices = Tensor::try_from(&vec![right_index as i64]).unwrap().to_device(device);
        (left_indices, right_indices)
    }

    //Performs a single move. Assumes that we only have a single rollout rn.
    pub fn perform_move(self, left_index : usize, right_index : usize) -> Self {
        let (left_indices, right_indices) = self.expand_single_indices(left_index, right_index);
        self.manual_step(&left_indices, &right_indices)
    }

    pub fn perform_move_diff(&self, left_index : usize, right_index : usize) -> RolloutStatesDiff {
        let (left_indices, right_indices) = self.expand_single_indices(left_index, right_index);
        self.perform_moves_diff(&left_indices, &right_indices)
    }

    pub fn manual_step(self, left_indices : &Tensor, right_indices : &Tensor) -> RolloutStates {
        let diff = self.perform_moves_diff(left_indices, right_indices);
        self.apply_diff(&diff)
    }

    ///Given indexing tensors of 1D shape R, with indices in 0..(k+t) for
    ///the left and right matrices to multiply for the next move, build
    ///a new RolloutStatesDiff representing the updates.
    pub fn perform_moves_diff(&self, left_indices : &Tensor, right_indices : &Tensor) -> RolloutStatesDiff {
        let _guard = no_grad_guard();

        let r = self.get_num_rollouts();
        let m = self.get_matrix_size();

        let left_indices = left_indices.reshape(&[r, 1, 1, 1]);
        let right_indices = right_indices.reshape(&[r, 1, 1, 1]);

        let expanded_shape = vec![r, 1, m, m];

        let left_indices = left_indices.expand(&expanded_shape, false);
        let right_indices = right_indices.expand(&expanded_shape, false);

        //Gets the left/right matrices which were sampled for the next step in rollouts
        //Dimensions R x M x M
        let left_matrices = self.matrices.gather(1, &left_indices, false);
        let right_matrices = self.matrices.gather(1, &right_indices, false);

        let left_matrices = left_matrices.reshape(&[r, m, m]);
        let right_matrices = right_matrices.reshape(&[r, m, m]);

        //R x M x M
        let added_matrices = left_matrices.matmul(&right_matrices);
        let added_matrices = added_matrices.reshape(&[r, 1, m, m]);

        //R x (M * M)
        let flattened_added_matrices = added_matrices.reshape(&[r, m * m]);

        //R x (M * M)
        let differences = &self.flattened_targets - &flattened_added_matrices;
        let squared_differences = differences.square();
        let distances = squared_differences.sum_dim_intlist(Some(&[1 as i64] as &[i64]), false, Kind::Float);

        let min_distances = self.min_distances.fmin(&distances);

        RolloutStatesDiff {
            min_distances,
            matrices : added_matrices,
        }
    }

    pub fn from_playout_bundle_initial_state(playout_bundle : &PlayoutBundle) -> RolloutStates {
        let _guard = no_grad_guard();

        let s = playout_bundle.matrix_bundle.flattened_initial_matrix_sets.size();
        let (n, k, m_squared) = (s[0], s[1], s[2]);
        let remaining_turns = playout_bundle.sketch_bundle.left_matrix_indices.size()[1] as usize;
        let m = (m_squared as f64).sqrt() as i64;

        let matrices = playout_bundle.matrix_bundle.flattened_initial_matrix_sets.reshape(&[n, k, m, m]);
        let flattened_targets = playout_bundle.matrix_bundle.flattened_matrix_targets.shallow_clone();

        //Now just need to derive the initial min_distances
        let reshaped_targets = flattened_targets.reshape(&[n, 1, m_squared]);

        //N x K x (M * M)
        let differences = &reshaped_targets - &playout_bundle.matrix_bundle.flattened_initial_matrix_sets;
        let squared_differences = differences.square();
        //N x K
        let distances = squared_differences.sum_dim_intlist(Some(&[1 as i64] as &[i64]), false, Kind::Float);
        //N
        let (min_distances, _) = distances.min_dim(1, false);

        RolloutStates {
            min_distances,
            flattened_targets,
            matrices,
            remaining_turns,
        }
    }

    //Expands a single-rollout "RolloutStates" to have R identical rollout states
    pub fn expand(self, R : usize) -> Self {
        let _guard = no_grad_guard();

        let matrices = self.matrices.expand(&[R as i64, -1, -1, -1], false); 
        let min_distances = self.min_distances.expand(&[R as i64], false);
        let flattened_targets = self.flattened_targets.expand(&[R as i64, -1], false);
        RolloutStates {
            min_distances,
            matrices,
            flattened_targets,
            remaining_turns : self.remaining_turns
        }
    }
}
