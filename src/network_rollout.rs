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

///Snapshot state of k^2 rollouts for each matrix product among an initial set of k matrices,
///where each snapshot takes place at some number of turns after the initial turn
pub struct NetworkRolloutState {
    ///1D tensor of size k^2, containing minimal distances along each rollout path
    pub min_distances : Tensor,
    ///The results of passing (source, target) matrix pairings through the injector_net
    ///for each of the k^2 playouts
    ///Dimension is  (k + t)  x  k^2 x F where "t" is the number of steps since the playout start
    pub single_embeddings : Vec<Tensor>,
    ///Flattened target matrix (dims 1 x m * m)
    pub flattened_target : Tensor,
    ///The matrices in each set, each of dims MxM, k^2 x (k+t) of 'em
    ///k^2 x (k + t) x M x M
    pub matrices : Tensor,
    ///The remaining number of turns
    pub remaining_turns : usize
}

impl NetworkRolloutState {
    pub fn step(self, network_config : &NetworkConfig) -> Self {
        //Really just need to update min_distances, single_embeddings, and matrices
        let _guard = no_grad_guard();
        
        //First, obtain policy matrices from all of the single_embeddings
        //k^2 x (k + t) x (k + t)
        let policy = network_config.get_policy(&self.single_embeddings);
        let k_squared = policy.size()[0];
        let k_plus_t = policy.size()[1];
        let m = self.matrices.size()[2];
        //Flatten the last dimensions of the policy matrices to prepare for sampling
        let policy_reshaped = policy.reshape(&[k_squared, k_plus_t * k_plus_t]);
        //Draw index samples from the policy matrix -- 1D shape k^2, 
        //indices are in (k + t) * (k + t)
        let sampled_joint_indices = policy_reshaped.multinomial(1, true).reshape(&[k_squared]);
        let left_indices = sampled_joint_indices.divide_scalar_mode(k_plus_t, "trunc");
        let right_indices = sampled_joint_indices.fmod(k_plus_t);

        let left_indices = left_indices.reshape(&[k_squared, 1, 1, 1]);
        let right_indices = right_indices.reshape(&[k_squared, 1, 1, 1]);

        let expanded_shape = vec![k_squared, 1, m, m];

        let left_indices = left_indices.expand(&expanded_shape, false);
        let right_indices = right_indices.expand(&expanded_shape, false);

        //Gets the left/right matrices which were sampled for the next step in rollouts
        //Dimensions k^2 x M x M
        let left_matrices = self.matrices.gather(1, &left_indices, false);
        let right_matrices = self.matrices.gather(1, &right_indices, false);

        let left_matrices = left_matrices.reshape(&[k_squared, m, m]);
        let right_matrices = right_matrices.reshape(&[k_squared, m, m]);

        //k^2 x M x M
        let added_matrices = left_matrices.matmul(&right_matrices);
        let added_matrices = added_matrices.reshape(&[k_squared, 1, m, m]);

        //k^2 x (M * M)
        let flattened_added_matrices = added_matrices.reshape(&[k_squared, m * m]);

        //k^2 x (M * M)
        let differences = &self.flattened_target - &flattened_added_matrices;
        let squared_differences = differences.square();
        let distances = squared_differences.sum_dim_intlist(&[1], false, Kind::Float);

        let min_distances = self.min_distances.fmin(&distances);

        let flattened_target_expanded = self.flattened_target.expand_as(&flattened_added_matrices);


        //k^2 x F
        let added_single_embeddings = network_config.injector_net.forward(&flattened_added_matrices, 
                                                                          &flattened_target_expanded);
        let mut single_embeddings = self.single_embeddings;
        single_embeddings.push(added_single_embeddings);

        //k^2 x (k + t + 1) x M x M
        let matrices = Tensor::concat(&[self.matrices, added_matrices], 1);

        let remaining_turns = self.remaining_turns - 1;
        NetworkRolloutState {
            remaining_turns,
            matrices,
            single_embeddings,
            min_distances,
            flattened_target : self.flattened_target
        }
    }

    pub fn new(game_state : GameState, network_config : &NetworkConfig) -> NetworkRolloutState {
        let _guard = no_grad_guard();

        let current_set_size = game_state.matrix_set.size(); 
        let current_distance = game_state.distance;
        let target = game_state.get_target();
        let remaining_turns = game_state.remaining_turns - 1;

        let mut starting_matrices = Vec::new();
        let mut starting_flattened_matrices = Vec::new();
        for i in 0..current_set_size {
            let matrix = game_state.matrix_set.get(i);
            let tensor = matrix_to_unbatched_tensor(matrix);
            starting_matrices.push(tensor);

            let flat_matrix = flatten_matrix(matrix);
            let flattened_tensor = vector_to_tensor(flat_matrix);
            starting_flattened_matrices.push(flattened_tensor);
        }

        //Dims: 1 x (M * M)
        let flattened_target = vector_to_tensor(flatten_matrix(target));

        //Dims: 1 x F
        let mut fat_starting_single_embeddings = network_config.get_single_embeddings(&starting_flattened_matrices, 
                                                                              &flattened_target);
        //The starting embeddings above have an extra "1" in their dimensions due to the
        //fact that we passed a single input to get_single_embeddings -- remove that extra dim
        let mut starting_single_embeddings = Vec::new();
        for fat_single_embedding in fat_starting_single_embeddings.drain(..) {
            let f = fat_single_embedding.size()[1];
            let single_embedding = fat_single_embedding.reshape(&[f]);
            starting_single_embeddings.push(single_embedding);
        }

        let mut min_distances = Vec::new();

        let mut single_embeddings = Vec::new();
        let mut matrices = Vec::new();

        for i in 0..current_set_size {
            let left_matrix = game_state.matrix_set.get(i);
            for j in 0..current_set_size {
                //TODO: We could do this with less communication overhead and better
                //GPU utilization by evaluating all pairwise matrix products on the GPU
                //Of course, to do so, we'll also need to compute distances on-device
                let right_matrix = game_state.matrix_set.get(j);
                let added_matrix = left_matrix.dot(&right_matrix);
                let flattened_matrix = flatten_matrix(added_matrix.view());
                let flattened_tensor = vector_to_tensor(flattened_matrix);
                
                let dist_to_added = sq_frob_dist(added_matrix.view(), game_state.target.view());
                let child_current_distance = current_distance.min(dist_to_added); 
                min_distances.push(child_current_distance);

                let mut single_embeddings_row = Vec::new();
                let mut matrices_row = Vec::new();

                //First, append the information from the starting sets, but as shallow clones
                for k in 0..current_set_size {
                    let matrix = starting_matrices[k].shallow_clone();
                    let single_embedding = starting_single_embeddings[k].shallow_clone();

                    matrices_row.push(matrix);
                    single_embeddings_row.push(single_embedding);
                }
                //Then, append the new matrix and the new embedding for the set
                matrices_row.push(matrix_to_unbatched_tensor(added_matrix.view()));
                let matrices_row = Tensor::stack(&matrices_row, 0);

                let added_embedding = network_config.injector_net.forward(&flattened_tensor, &flattened_target); 

                //Need to flatten the added embedding because batch size was just one
                let f = added_embedding.size()[1];
                let added_embedding = added_embedding.reshape(&[f]);

                single_embeddings_row.push(added_embedding);

                //(k + 1) x F
                let single_embeddings_row = Tensor::stack(&single_embeddings_row, 0);

                single_embeddings.push(single_embeddings_row);
                matrices.push(matrices_row);
            }
        }
        //k^2 x (k + t) x M x M
        let matrices = Tensor::stack(&matrices, 0);

        //(k + 1) x k^2 x F
        let single_embeddings = Tensor::stack(&single_embeddings, 1);
        //(k + 1)  x  k^2 x F
        let single_embeddings = single_embeddings.unbind(0);

        let min_distances = Tensor::try_from(&min_distances).unwrap();

        NetworkRolloutState {
            single_embeddings,
            matrices,
            remaining_turns,
            min_distances,
            flattened_target
        }
    }
}
