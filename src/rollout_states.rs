use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use crate::network::*;
use crate::neural_utils::*;
use crate::game_state::*;
use crate::array_utils::*;
use crate::params::*;
use crate::training_examples::*;
use crate::network_config::*;
use ndarray::*;
use std::convert::TryFrom;

use rand::Rng;
use rand::seq::SliceRandom;
use core::iter::Sum;


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

impl RolloutStates {
    pub fn get_num_rollouts(&self) -> i64 {
        self.matrices.size()[0]
    }
    pub fn get_matrix_size(&self) -> i64 {
        self.matrices.size()[2]
    }
    pub fn get_num_matrices(&self) -> i64 {
        self.matrices.size()[1]
    }

    ///Given indexing tensors of 1D shape R, with indices in 0..(k+t) for
    ///the left and right matrices to multiply for the next move, build
    ///a new RolloutStates representing the updates. Also return a copy of
    ///the flattened added matrices tensor
    pub fn perform_moves(self, left_indices : &Tensor, right_indices : &Tensor) -> (Self, Tensor) {
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
        let distances = squared_differences.sum_dim_intlist(&[1], false, Kind::Float);

        let min_distances = self.min_distances.fmin(&distances);

        //R x (k + t + 1) x M x M
        let matrices = Tensor::concat(&[self.matrices, added_matrices], 1);

        let remaining_turns = self.remaining_turns - 1;
        
        let updated_states = RolloutStates {
            min_distances,
            flattened_targets : self.flattened_targets,
            matrices,
            remaining_turns
        };
        (updated_states, flattened_added_matrices)
    }
    pub fn from_playout_bundle_initial_state(playout_bundle : &PlayoutBundle) -> RolloutStates {
        let s = playout_bundle.flattened_initial_matrix_sets.size();
        let (n, k, m_squared) = (s[0], s[1], s[2]);
        let remaining_turns = playout_bundle.left_matrix_indices.size()[1] as usize;
        let m = (m_squared as f64).sqrt() as i64;

        let matrices = playout_bundle.flattened_initial_matrix_sets.reshape(&[n, k, m, m]);
        let flattened_targets = playout_bundle.flattened_matrix_targets.shallow_clone();

        //Now just need to derive the initial min_distances
        let reshaped_targets = flattened_targets.reshape(&[n, 1, m_squared]);

        //N x K x (M * M)
        let differences = &reshaped_targets - &playout_bundle.flattened_initial_matrix_sets;
        let squared_differences = differences.square();
        //N x K
        let distances = squared_differences.sum_dim_intlist(&[1], false, Kind::Float);
        //N
        let (min_distances, _) = distances.min_dim(1, false);

        RolloutStates {
            min_distances,
            flattened_targets,
            matrices,
            remaining_turns,
        }
    }

    ///Constructs a single-rollout "RolloutStates" from a given game-state.
    pub fn from_single_game_state(game_state : GameState) -> RolloutStates {
        let k = game_state.matrix_set.len();
        let current_distance = game_state.distance;
        let target = game_state.get_target();
        let remaining_turns = game_state.remaining_turns;

        let mut matrices = Vec::new();
        for i in 0..k {
            let matrix = game_state.matrix_set.get(i);
            let tensor = matrix_to_unbatched_tensor(matrix);
            matrices.push(tensor);
        }
        //1xKxMxM
        let matrices = Tensor::stack(&matrices, 0).unsqueeze(0);

        //1 x (M * M)
        let flattened_targets = vector_to_tensor(flatten_matrix(target));

        //Only one rollout
        let min_distances = Tensor::of_slice(&[current_distance]);

        RolloutStates {
            min_distances,
            flattened_targets,
            matrices,
            remaining_turns
        }
    }
    //Expands a single-rollout "RolloutStates" to have R identical rollout states
    pub fn expand(self, R : usize) -> Self {
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
