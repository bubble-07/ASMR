use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use crate::network::*;
use crate::neural_utils::*;
use crate::game_state::*;
use crate::array_utils::*;
use crate::params::*;
use crate::training_examples::*;
use crate::network_config::*;
use crate::rollout_states::*;
use crate::visit_logit_matrices::*;
use crate::network_module::*;
use ndarray::*;
use std::convert::TryFrom;
use std::iter::zip;

use rand::Rng;
use rand::seq::SliceRandom;
use core::iter::Sum;
use crate::peeling_states::*;

///Snapshot state of R rollouts for each matrix product among an initial set of k matrices,
///where each snapshot takes place at some number of turns after the initial turn
pub struct NetworkRolloutState {
    ///Non-network-based states of the R rollouts
    pub rollout_states : RolloutStates,

    ///Ordered per-layer peeling states. L of them, each with N->R, K->(k+t) for
    ///t the current number of elapsed turns.
    pub peeling_states : PeelLayerStates,

    ///Output activation maps for all of the matrices in the network
    ///(k + t) x R x F
    pub output_activations : Tensor,

    ///Global output activation maps
    ///R x F
    pub global_output_activation : Tensor,

    ///Logits for child visit tensors
    ///R x (k + t) x (k + t)
    pub child_visit_logits : VisitLogitMatrices,
}

///Diff to a rollout state with R rollouts 
pub struct NetworkRolloutDiff {
    ///Diff to the rollout states
    pub rollout_states_diff : RolloutStatesDiff,
    ///States of the peel track added
    pub peel_track_state : PeelTrackStates,
    ///Dim RxF - output activation of chosen matrix
    pub added_output_activations : Tensor,
    ///Leftover logits for the child visit logit matrix
    ///Dim Rx(k+t_init)
    pub left_logits : Tensor,
    ///Dim Rx(K+t_init + 1)
    pub right_logits : Tensor,
    pub left_indices : Tensor,
    pub right_indices : Tensor,
}

impl NetworkRolloutDiff {
    ///Assuming that this network rollout diff has a bunch of different
    ///rollouts, splits it into single-rollout diffs
    pub fn split_to_singles(self) -> Vec<NetworkRolloutDiff> {
        let rollout_states_diffs = self.rollout_states_diff.split_to_singles();
        let peel_track_states = self.peel_track_state.split_to_singles();
        let added_output_activations = self.added_output_activations.split(1, 0);
        let left_logits = self.left_logits.split(1, 0);
        let right_logits = self.right_logits.split(1, 0);
        let left_indices = self.left_indices.split(1, 0);
        let right_indices = self.right_indices.split(1, 0);

        zip(zip(zip(zip(zip(zip(rollout_states_diffs, peel_track_states), added_output_activations),
            left_logits), right_logits), left_indices), right_indices)
        .map(|((((((rollout_states_diff, peel_track_state), added_output_activations),
              left_logits), right_logits), left_indices), right_indices)| {
            NetworkRolloutDiff {
                rollout_states_diff,
                peel_track_state,
                added_output_activations,
                left_logits,
                right_logits,
                left_indices,
                right_indices,
            }
        }).collect()
    }
}

impl NetworkRolloutState {
    pub fn shallow_clone(&self) -> Self {
        Self {
            rollout_states : self.rollout_states.shallow_clone(),
            peeling_states : self.peeling_states.shallow_clone(),
            output_activations : self.output_activations.shallow_clone(),
            global_output_activation : self.global_output_activation.shallow_clone(),
            child_visit_logits : self.child_visit_logits.shallow_clone(),
        }
    }
    ///Assuming that we currently have only one rollout, expands to incorporate
    ///num_matrices * num_matrices rollouts
    pub fn expand_for_all_moves(self) -> Self {
        let num_matrices = self.rollout_states.get_num_matrices();
        let num_children = num_matrices * num_matrices;

        self.expand(num_children as usize)
    }
    ///Assuming that this rollout state currently only contains one rollout,
    ///computes all diffs to the current rollout state which cover all of the
    ///subsequent next possible moves
    pub fn diff_all_moves(&self, network_config : &NetworkConfig) -> NetworkRolloutDiff {
        let device = self.rollout_states.matrices.device();
        let num_matrices = self.rollout_states.get_num_matrices();

        let (left_indices, right_indices) = generate_2d_index_tensor_span(num_matrices, device);

        let expanded_clone = self.shallow_clone().expand_for_all_moves();
        let result = expanded_clone.manual_step_diff(network_config, &left_indices, &right_indices);
        result
    }
    ///Assuming that this rollout state currently only contains one rollout,
    ///expands it to rollout states which cover all of the subsequent
    ///next possible moves
    pub fn perform_all_moves(self, network_config : &NetworkConfig) -> Self {
        let diff = self.diff_all_moves(network_config);
        let result = self.expand_for_all_moves();
        result.apply_diff(&diff)
    }

    //Expands a single-rollout NetworkRolloutState to have R identical rollouts
    pub fn expand(self, R : usize) -> Self {
        let _guard = no_grad_guard();

        let rollout_states = self.rollout_states.expand(R);
        let peeling_states = self.peeling_states.expand(R);
        let child_visit_logits = self.child_visit_logits.expand(R);

        let R = R as i64;

        let output_activations = self.output_activations.expand(&[-1, R, -1], false);
        let global_output_activation = self.global_output_activation.expand(&[R, -1], false);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation,
            child_visit_logits,
        }
    }
    ///Splits to a collection of network rollout states
    ///where each rollout state object contains the specified number
    ///of samples [according to split_sizes]
    pub fn split(self, split_sizes : &[i64]) -> Vec<NetworkRolloutState> {
        let rollout_states = self.rollout_states.split(split_sizes);
        let peeling_states = self.peeling_states.split(split_sizes);
        let child_visit_logits = self.child_visit_logits.split(split_sizes);

        let output_activations = self.output_activations.split_with_sizes(split_sizes, 1);
        let global_output_activation = self.global_output_activation.split_with_sizes(split_sizes, 0);

        zip(zip(zip(zip(rollout_states, peeling_states), child_visit_logits),
            output_activations), global_output_activation)
        .map(|((((rollout_states, peeling_states), child_visit_logits),
            output_activations), global_output_activation)|
            NetworkRolloutState {
                rollout_states,
                peeling_states,
                child_visit_logits,
                output_activations,
                global_output_activation
            }
        )
        .collect()
    }

    ///Merges a collection of rollout states with the same values
    ///of k, t, m and f, but possibly differing numbers of
    ///rollouts or differing numbers of remaining turns.
    ///(the output number of remaining turns will be the maximum of
    ///all inputs)
    pub fn merge(mut states : Vec<NetworkRolloutState>) -> NetworkRolloutState {
        let mut rollout_states = Vec::new();
        let mut peeling_states = Vec::new();
        let mut output_activations = Vec::new();
        let mut global_output_activation = Vec::new();
        let mut child_visit_logits = Vec::new();
        for state in states.drain(..) {
            rollout_states.push(state.rollout_states);
            peeling_states.push(state.peeling_states);
            output_activations.push(state.output_activations);
            global_output_activation.push(state.global_output_activation);
            child_visit_logits.push(state.child_visit_logits);
        }
        let rollout_states = RolloutStates::merge(rollout_states);
        let peeling_states = PeelLayerStates::merge(peeling_states);
        let output_activations = Tensor::concat(&output_activations, 1);
        let global_output_activation = Tensor::concat(&global_output_activation, 0);
        let child_visit_logits = VisitLogitMatrices::merge(child_visit_logits);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation,
            child_visit_logits,
        }
    }
    pub fn step(self, network_config : &NetworkConfig) -> Self {
        let (left_indices, right_indices) = self.child_visit_logits.draw_indices();
        
        self.manual_step(network_config, &left_indices, &right_indices)
    }

    pub fn complete_network_rollouts(self, network_config : &NetworkConfig) -> Self {
        let mut result = self;
        while result.rollout_states.remaining_turns > 0 {
            result = result.step(network_config);
        }
        result
    }
    
    pub fn manual_step_diff(&self, network_config : &NetworkConfig,
                       left_indices : &Tensor, right_indices : &Tensor) -> NetworkRolloutDiff {
        let init_k_plus_t = self.rollout_states.get_num_matrices() as usize;

        //Compute diff in rollout states
        let rollout_states_diff = self.rollout_states.perform_moves_diff(left_indices, right_indices);

        //Compute diff in input embeddings
        let flattened_added_matrices = rollout_states_diff.get_flattened_added_matrices();
        let flattened_transformed_added_matrices = flattened_added_matrices.asinh().detach();

        //R x F
        let added_single_embeddings = network_config.injector_net.forward(&flattened_transformed_added_matrices);

        //Compute diff in network activations
        
        //Update to the new layer activation peeling states by
        //"peeling forward" through the peel network, also yielding the
        //final activation out of the peeling track
        let (peel_track_state, added_output_activations) = 
            network_config.peel_net.peel_forward_diff(&self.peeling_states, &added_single_embeddings);

        //Expand child visit logit matrices from the newly-added output activations
        let mut left_logits = Vec::new();
        let mut right_logits = Vec::new();
        for other_index in 0..init_k_plus_t {
            let other_activations = self.output_activations.i((other_index as i64, .., ..));
            //Other as the left index
            //Dimension Rx1
            let left_logit = network_config.policy_extraction_net.forward(&other_activations, 
                                                                           &added_output_activations, 
                                                                           &self.global_output_activation);
            //Other as the right index
            let right_logit = network_config.policy_extraction_net.forward(&added_output_activations,
                                                                           &other_activations,
                                                                           &self.global_output_activation);
            left_logits.push(left_logit);
            right_logits.push(right_logit);
        }

        let corner_logit = network_config.policy_extraction_net.forward(&added_output_activations,
                                                                         &added_output_activations,
                                                                         &self.global_output_activation);
        right_logits.push(corner_logit);

        //Dimension Rx(k+t_init)
        let left_logits = Tensor::concat(&left_logits, 1);
        //Dimension Rx(k+t_init + 1)
        let right_logits = Tensor::concat(&right_logits, 1);

        let left_indices = left_indices.shallow_clone();
        let right_indices = right_indices.shallow_clone();

        NetworkRolloutDiff {
            rollout_states_diff,
            peel_track_state,
            added_output_activations,
            left_logits,
            right_logits,
            left_indices,
            right_indices,
        }
    }

    pub fn apply_diff(self, network_rollout_diff : &NetworkRolloutDiff) -> Self {
        //Update rollout states
        let rollout_states = self.rollout_states.apply_diff(&network_rollout_diff.rollout_states_diff);
        
        //Update peeling tracks
        let peeling_states = self.peeling_states.push_tracks(&network_rollout_diff.peel_track_state);

        //Mask the chosen indices in the child visit logits
        let mut child_visit_logits = self.child_visit_logits;
        child_visit_logits.mask_chosen(&network_rollout_diff.left_indices,
                                       &network_rollout_diff.right_indices);

        //Reshape so that we can properly append
        //Varies across first coordinate, so should be column vectors
        let left_logits = network_rollout_diff.left_logits.unsqueeze(2);
        //Varies across second coordinate, so should be row vectors
        let right_logits = network_rollout_diff.right_logits.unsqueeze(1);

        let child_visit_logits = Tensor::concat(&[child_visit_logits.0, left_logits], 2);
        let child_visit_logits = Tensor::concat(&[child_visit_logits, right_logits], 1);
        let child_visit_logits = VisitLogitMatrices(child_visit_logits);
        
        //Add output activations to our running list
        let added_output_activations = network_rollout_diff.added_output_activations.unsqueeze(0);
        let output_activations = Tensor::concat(&[self.output_activations, added_output_activations], 0);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation : self.global_output_activation,
            child_visit_logits,
        }
    }

    //Performs a single move. Assumes that we only have a single rollout rn.
    pub fn perform_move(self, network_config : &NetworkConfig, left_index : usize, right_index : usize) -> Self {
        let left_indices = Tensor::try_from(&vec![left_index as i64]).unwrap();
        let right_indices = Tensor::try_from(&vec![right_index as i64]).unwrap();
        self.manual_step(network_config, &left_indices, &right_indices)
    }

    pub fn manual_step(self, network_config : &NetworkConfig, 
                       left_indices : &Tensor, right_indices : &Tensor) -> Self {
        let diff = self.manual_step_diff(network_config, left_indices, right_indices);
        self.apply_diff(&diff)
    }

    //From an initial rolloutstates - must not have made any turns yet!
    pub fn from_rollout_states(network_config : &NetworkConfig, rollout_states : RolloutStates)
        -> NetworkRolloutState {
        let s = rollout_states.matrices.size();
        let (r, k_plus_t, m, _) = (s[0], s[1], s[2], s[3]);

        //Using asinh for mapping due to having logarithmic growth
        //with respect to the absolute value of the argument, which is roughly
        //what we'd desire for normalizing the effects of repeated matrix multiplication
        let mut flattened_transformed_matrix_sets = rollout_states.matrices
                                                .reshape(&[r, k_plus_t, m * m])
                                                .asinh().detach()
                                                .unbind(1);
        let transformed_flattened_targets = rollout_states.flattened_targets.asinh().detach();

        //Concat on the targets, since the main net will expect the target to be the last matrix
        flattened_transformed_matrix_sets.push(transformed_flattened_targets);


        let single_embeddings = network_config.get_single_embeddings(&flattened_transformed_matrix_sets);

        let (layer_activations, global_output_activation, output_activations) =
            network_config.get_main_net_outputs(&single_embeddings);

        let child_visit_logits = network_config.get_policy_logits(&global_output_activation,
                                                                  &output_activations);
        let child_visit_logits = VisitLogitMatrices(child_visit_logits);

        let peeling_states = network_config.peel_net.forward_to_peel_state(&layer_activations);

        let output_activations = Tensor::stack(&output_activations, 0);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation,
            child_visit_logits,
        }
    }
}
