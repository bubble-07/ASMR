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

use rand::Rng;
use rand::seq::SliceRandom;
use core::iter::Sum;

///Snapshot state of R rollouts for each matrix product among an initial set of k matrices,
///where each snapshot takes place at some number of turns after the initial turn
pub struct NetworkRolloutState {
    ///Non-network-based states of the R rollouts
    pub rollout_states : RolloutStates,

    ///Ordered per-layer peeling states. L of them, each with N->R, K->(k+t) for
    ///t the current number of elapsed turns.
    pub peeling_states : Vec<PeelLayerState>,

    ///Output activation maps for all of the matrices in the network
    ///(k + t) x R x F
    pub output_activations : Vec<Tensor>,

    ///Global output activation maps
    ///R x F
    pub global_output_activation : Tensor,

    ///Logits for child visit tensors
    ///R x (k + t) x (k + t)
    pub child_visit_logits : VisitLogitMatrices,

    ///Transformed added targets
    pub transformed_flattened_targets : Tensor,
}

impl NetworkRolloutState {
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
            network_config.peel_net.peel_forward(&self.peeling_states, &added_single_embeddings);

        //Expand child visit logit matrices from the newly-added output activations
        let mut left_logits = Vec::new();
        let mut right_logits = Vec::new();
        for other_index in 0..init_k_plus_t {
            let other_activations = &self.output_activations[other_index];
            //Other as the left index
            //Dimension Rx1
            let left_logit = network_config.policy_extraction_net.forward(other_activations, 
                                                                           &added_output_activations, 
                                                                           &self.global_output_activation);
            //Other as the right index
            let right_logit = network_config.policy_extraction_net.forward(&added_output_activations,
                                                                           other_activations,
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
        let mut output_activations = self.output_activations;
        output_activations.push(added_output_activations);

        NetworkRolloutState {
            rollout_states,
            peeling_states,
            output_activations,
            global_output_activation : self.global_output_activation,
            child_visit_logits,
            transformed_flattened_targets : self.transformed_flattened_targets,
        }
    }

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
