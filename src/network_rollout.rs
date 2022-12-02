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

    ///Transformed added targets
    ///R x M
    pub transformed_flattened_targets : Tensor,
}

impl NetworkRolloutState {
    ///Assuming that this rollout state currently only contains one rollout,
    ///expands it to rollout states which cover all of the subsequent
    ///next possible moves
    pub fn perform_all_moves(self, network_config : &NetworkConfig) -> Self {
        let device = self.rollout_states.matrices.device();
        let num_matrices = self.rollout_states.get_num_matrices();
        let num_children = num_matrices * num_matrices;

        //Construct the left/right indices tensors
        let left_indices = Tensor::arange(num_matrices, (Kind::Int64, device));
        let left_indices = left_indices.repeat(&[num_matrices]);

        let right_indices = Tensor::arange(num_matrices, (Kind::Int64, device));
        let right_indices = right_indices.repeat_interleave_self_int(num_matrices, Option::None,
                                                                     Option::Some(num_children));

        //Expand and perform moves
        let result = self.expand(num_children as usize);
        let result = result.manual_step(&network_config, &left_indices, &right_indices);
        result
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
        let transformed_flattened_targets = self.transformed_flattened_targets
                                                .expand(&[R, -1], false);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation,
            child_visit_logits,
            transformed_flattened_targets,
        }
    }
    ///Splits to a collection of network rollout states
    ///where each rollout state object contains the specified number
    ///of samples [according to split_sizes]
    pub fn split(self, split_sizes : &[i64]) -> Vec<NetworkRolloutState> {
        let rollout_states = self.rollout_states.split(split_sizes);
        let peeling_states = self.peeling_states.split(split_sizes);
        let child_visit_logits = self.child_visit_logits.split(split_sizes);

        let transformed_flattened_targets = self.transformed_flattened_targets.split_with_sizes(split_sizes, 0);         
        let output_activations = self.output_activations.split_with_sizes(split_sizes, 1);
        let global_output_activation = self.global_output_activation.split_with_sizes(split_sizes, 0);

        zip(zip(zip(zip(zip(rollout_states, peeling_states), child_visit_logits),
            transformed_flattened_targets), output_activations), global_output_activation)
        .map(|(((((rollout_states, peeling_states), child_visit_logits),
            transformed_flattened_targets), output_activations), global_output_activation)|
            NetworkRolloutState {
                rollout_states,
                peeling_states,
                child_visit_logits,
                transformed_flattened_targets,
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
        let mut transformed_flattened_targets = Vec::new();
        for state in states.drain(..) {
            rollout_states.push(state.rollout_states);
            peeling_states.push(state.peeling_states);
            output_activations.push(state.output_activations);
            global_output_activation.push(state.global_output_activation);
            child_visit_logits.push(state.child_visit_logits);
            transformed_flattened_targets.push(state.transformed_flattened_targets);
        }
        let rollout_states = RolloutStates::merge(rollout_states);
        let peeling_states = PeelLayerStates::merge(peeling_states);
        let output_activations = Tensor::concat(&output_activations, 1);
        let global_output_activation = Tensor::concat(&global_output_activation, 0);
        let child_visit_logits = VisitLogitMatrices::merge(child_visit_logits);
        let transformed_flattened_targets = Tensor::concat(&transformed_flattened_targets, 0);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation,
            child_visit_logits,
            transformed_flattened_targets,
        }
    }
    pub fn step(self, network_config : &NetworkConfig) -> Self {
        let (left_indices, right_indices) = self.child_visit_logits.draw_indices();
        
        self.manual_step(network_config, &left_indices, &right_indices)
    }

    pub fn manual_step(self, network_config : &NetworkConfig, 
                       left_indices : &Tensor, right_indices : &Tensor) -> Self {
        let init_k_plus_t = self.rollout_states.get_num_matrices() as usize;

        //First, perform the move
        let (rollout_states, flattened_added_matrices) = 
            self.rollout_states.perform_moves(left_indices, right_indices);

        let flattened_transformed_added_matrices = flattened_added_matrices.asinh().detach();

        //Now to keep the network state up-to-date
        
        //Ensure that we mark the chosen indices as visited in the logit matrix
        let mut child_visit_logits = self.child_visit_logits;
        child_visit_logits.mask_chosen(left_indices, right_indices);
        
        //Derive the pre-activation for a new peel track
        let transformed_flattened_targets = &self.transformed_flattened_targets;

        //R x F
        let added_single_embeddings = network_config.injector_net.forward(&flattened_transformed_added_matrices,
                                                                          transformed_flattened_targets);
        
        //Update to the new layer activation peeling states by
        //"peeling forward" through the peel network, also yielding the
        //final activation out of the peeling track
        let (peeling_states, added_output_activations) = 
            network_config.peel_net.peel_forward(self.peeling_states, &added_single_embeddings);

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

        //Reshape so that we can properly append
        //Varies across first coordinate, so should be column vectors
        let left_logits = left_logits.unsqueeze(2);
        //Varies across second coordinate, so should be row vectors
        let right_logits = right_logits.unsqueeze(1);

        let child_visit_logits = Tensor::concat(&[child_visit_logits.0, left_logits], 2);
        let child_visit_logits = Tensor::concat(&[child_visit_logits, right_logits], 1);
        let child_visit_logits = VisitLogitMatrices(child_visit_logits);
        
        //Add output activations to our running list
        let added_output_activations = added_output_activations.unsqueeze(0);
        let output_activations = Tensor::concat(&[self.output_activations, added_output_activations], 0);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation : self.global_output_activation,
            child_visit_logits,
            transformed_flattened_targets : self.transformed_flattened_targets,
        }
    }

    //From an initial rolloutstates - must not have made any turns yet!
    pub fn from_rollout_states(network_config : &NetworkConfig, rollout_states : RolloutStates)
        -> NetworkRolloutState {
        let s = rollout_states.matrices.size();
        let (r, k_plus_t, m, _) = (s[0], s[1], s[2], s[3]);

        //Using asinh for mapping due to having logarithmic growth
        //with respect to the absolute value of the argument, which is roughly
        //what we'd desire for normalizing the effects of repeated matrix multiplication
        let flattened_transformed_matrix_sets = rollout_states.matrices
                                                .reshape(&[r, k_plus_t, m * m])
                                                .asinh().detach()
                                                .unbind(1);
        let transformed_flattened_targets = rollout_states.flattened_targets.asinh().detach();


        let single_embeddings = network_config.get_single_embeddings(&flattened_transformed_matrix_sets,
                                                                     &transformed_flattened_targets);

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
            transformed_flattened_targets,
        }
    }
}
