use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};
use crate::training_examples::*;
use crate::network_rollout::*;
use crate::network_config::*;
use crate::rollout_states::*;
use crate::params::*;
use std::iter::zip;

struct PlayoutBundleTailNode {
    //Tensor of dims Nx([K+i]*[K+i])
    pub child_visit_probabilities : Tensor,
    //Tensor of dim N
    pub left_matrix_indices : Tensor,
    //Tensor of dim N
    pub right_matrix_indices : Tensor
}

//Just the trailing part of a playout bundle, after
//the initial set [and visit probability matrix] is done with
struct PlayoutBundleTail {
    tail_nodes : Vec<PlayoutBundleTailNode>,
}

impl PlayoutBundleTail {
    fn len(&self) -> usize {
        self.tail_nodes.len()
    }
    fn pop(&mut self) -> Option<PlayoutBundleTailNode> {
        self.tail_nodes.pop()
    }
    fn new(playout_bundle : PlayoutBundle) -> PlayoutBundleTail {
        //We're going to reverse each involved list, since they're in
        //chronological order, but we need them to _pop_ in chronological order
        let mut child_visit_probabilities = playout_bundle.child_visit_probabilities;
        child_visit_probabilities.reverse();
        //Need to remove the first child visit probability matrix, since that's
        //handled in the base.
        child_visit_probabilities.pop();

        //Need to remove the last indices, because these are used to get to the final
        //state, and so they're not necessary to get a rollout state which is necessary
        //for training.
        let mut left_matrix_indices = playout_bundle.left_matrix_indices.unbind(1);
        let mut right_matrix_indices = playout_bundle.right_matrix_indices.unbind(1);
        left_matrix_indices.pop();
        right_matrix_indices.pop();
        left_matrix_indices.reverse();
        right_matrix_indices.reverse();

        let tail_nodes : Vec<PlayoutBundleTailNode> = 
            zip(zip(child_visit_probabilities, left_matrix_indices), right_matrix_indices)
            .map(|((child_visit_probabilities, left_matrix_indices), right_matrix_indices)|
            PlayoutBundleTailNode {
                child_visit_probabilities,
                left_matrix_indices,
                right_matrix_indices,
            }).collect();
        PlayoutBundleTail {
            tail_nodes
        }
    }
}

struct BundleState {
    pub init_set_size : usize,
    pub final_set_size : usize,
    pub weight : f64,
    pub network_rollout_state : NetworkRolloutState,
    pub bundle_tail : PlayoutBundleTail,
}

impl BundleState {
    pub fn get_batch_size(&self) -> i64 {
        self.network_rollout_state.global_output_activation.size()[0]
    }
    pub fn get_device(&self) -> Device {
        self.network_rollout_state.output_activations.device()
    }
    pub fn has_remaining_moves(&self) -> bool {
        self.bundle_tail.len() > 0
    }
    ///Given bundle-states which all have the same current set size
    ///(implicitly referenced in their NetworkRolloutState), advance
    ///them forward to obtain a peel loss and new bundle states one turn
    ///into the future.
    pub fn advance(network_config : &NetworkConfig, params : &Params,
                   mut bundle_states : Vec<BundleState>) -> (Tensor, Vec<BundleState>) {
        let device = bundle_states[0].get_device();
        let sizes : Vec<i64> = bundle_states.iter().map(|x| x.get_batch_size()).collect();
        let mut total_loss = Tensor::zeros(&[], (Kind::Float, device));

        let mut left_matrix_indices = Vec::new();
        let mut right_matrix_indices = Vec::new();
        let mut network_rollout_states = Vec::new();
        let mut carryforwards = Vec::new();
        for mut bundle_state in bundle_states.drain(..) {
           let tail_node = bundle_state.bundle_tail.pop().unwrap();
           left_matrix_indices.push(tail_node.left_matrix_indices);
           right_matrix_indices.push(tail_node.right_matrix_indices);

           let carryforward = (bundle_state.init_set_size, bundle_state.final_set_size,
                           bundle_state.weight, bundle_state.bundle_tail,
                           tail_node.child_visit_probabilities);
           carryforwards.push(carryforward);

           network_rollout_states.push(bundle_state.network_rollout_state);
        }

        //Merge all relevant state for moving forward
        let left_matrix_indices = Tensor::concat(&left_matrix_indices, 0);
        let right_matrix_indices = Tensor::concat(&right_matrix_indices, 0);
        let network_rollout_states = NetworkRolloutState::merge(network_rollout_states);

        //Move forward
        let network_rollout_states = network_rollout_states.manual_step(network_config, &left_matrix_indices,
                                                                        &right_matrix_indices);
        
        let mut result_states = Vec::new();

        //And finally, split the result back out and calculate loss
        let mut network_rollout_states = network_rollout_states.split(&sizes);
        for (network_rollout_state, carryforwards) in network_rollout_states.drain(..)
                                                    .zip(carryforwards.drain(..)) {
            let (init_set_size, final_set_size,
                 weight, bundle_tail, child_visit_probabilities) = carryforwards;

            let network_visit_logits = &network_rollout_state.child_visit_logits;
            let peel_loss = network_visit_logits.get_peel_loss(&child_visit_probabilities);
            total_loss += weight * peel_loss;
            
            let result_state = BundleState {
                init_set_size,
                final_set_size,
                weight,
                bundle_tail,
                network_rollout_state
            };
            result_states.push(result_state);

        }

        (total_loss, result_states)
    }
}

pub fn get_loss_for_playout_bundles(network_config : &NetworkConfig, params : &Params,
                                    mut playout_bundles : Vec<(f64, PlayoutBundle)>) -> Tensor {
    //SETUP

    let mut total_loss = Tensor::zeros(&[], (Kind::Float, playout_bundles[0].1.device()));
    let num_bundles = playout_bundles.len();
    let mut bundle_states : Vec<Option<BundleState>> = std::iter::repeat_with(|| Option::None)
                                                       .take(num_bundles).collect();

    //BASE LOSS

    //First, take all of the playout bundles, and merge the starting
    //configurations for those with the same initial set size temporarily,
    //splitting those back into network rollouts once processed.
    //Losses on the initial sets will be recorded at this point
    //Find the minimum and maximum initial set sizes
    let min_initial_set_size = playout_bundles.iter().map(|(_, b)| b.get_init_set_size()).min().unwrap();
    let max_initial_set_size = playout_bundles.iter().map(|(_, b)| b.get_init_set_size()).max().unwrap();
    let initial_set_size_spread = max_initial_set_size - min_initial_set_size;

    //Populate an array with one element per each initial set size,
    //where each element is a vector containing indices of weighted
    //playout bundles which have that initial set size
    let mut init_set_size_index_map : Vec<Vec<usize>> = vec![Vec::new(); initial_set_size_spread + 1];
    for i in 0..num_bundles {
        let playout_bundle_size = playout_bundles[i].1.get_init_set_size();
        let set_size_offset = playout_bundle_size - min_initial_set_size;
        init_set_size_index_map[set_size_offset].push(i);
    }

    //Need to be able to move elements from this in a scattered way
    let mut playout_bundles : Vec<Option<(f64, PlayoutBundle)>>
          = playout_bundles.drain(..).map(|x| Option::Some(x)).collect();

    //Loop through each possible initial set size, and take the playout bundles with
    //those indices to construct a new NetworkRolloutState, which we then
    //use to create BundleStates for the bundle_states
    for (set_size_offset, mut indices_for_size) in init_set_size_index_map.drain(..).enumerate() {
        let mut bundles_for_size = Vec::new();
        let mut weights_for_size = Vec::new();
        for index in &indices_for_size {
            let (weight, bundle) = std::mem::replace(&mut playout_bundles[*index], Option::None).unwrap();
            bundles_for_size.push(bundle);
            weights_for_size.push(weight);
        }
        //With all of the bundles of the given size together, make a combined network rollout state
        let mut rollout_states_for_size = Vec::new();
        for bundle in &bundles_for_size {
            let rollout_states = RolloutStates::from_playout_bundle_initial_state(bundle);
            rollout_states_for_size.push(rollout_states);
        }
        //Determine the sizes of all of the rollout states
        let batch_sizes : Vec<i64> = rollout_states_for_size.iter().map(|x| x.get_num_rollouts()).collect();

        let merged_rollout_state = RolloutStates::merge(rollout_states_for_size);
        let merged_network_rollout = NetworkRolloutState::from_rollout_states(network_config, merged_rollout_state);

        //Then split that into a bunch of network rollout states for each bundle
        let mut network_rollouts = merged_network_rollout.split(&batch_sizes);

        for (((index, weight), bundle), network_rollout_state) in
            zip(zip(zip(indices_for_size.drain(..), weights_for_size.drain(..)), 
                    bundles_for_size.drain(..)), network_rollouts.drain(..)) {
            //Compute the base loss, and add it to our total
            let network_visit_logits = &network_rollout_state.child_visit_logits;
            let actual_visit_probabilities = &bundle.child_visit_probabilities[0];
            let base_loss = network_visit_logits.get_loss(actual_visit_probabilities);
            total_loss += weight * base_loss;

            //Get the newly-minted bundle to package
            let init_set_size = bundle.get_init_set_size();
            let final_set_size = bundle.get_final_set_size();
            //Note: Some of these bundles might have empty tails.
            //That's okay, we'll remove 'em prior to handling peels.
            let bundle_tail = PlayoutBundleTail::new(bundle);
            let bundle_state = BundleState {
                init_set_size,
                final_set_size,
                weight,
                network_rollout_state, 
                bundle_tail
            };
            //And put it in the right place
            bundle_states[index] = Option::Some(bundle_state);
        }
    }
    //All items in bundle_states should be populated

    //PEEL LOSS

    //Then, starting from the smallest initial set size, count up in
    //set-sizes, folding in rollout states for
    //all stored playout bundles which have
    //the current set-size as part of their interval.
    //After each iteration, the updated network states are
    //placed back into the backing map for these, and partial loss
    //obtained during this process will be accumulated.
    
    let max_final_set_size = bundle_states.iter().map(|x| x.as_ref().unwrap().final_set_size).max().unwrap();
    let min_peel_set_size = min_initial_set_size + 1;
    let peel_set_size_spread = max_final_set_size - min_peel_set_size;

    //Populate an array with one element per each possible peel set size
    //where the elements are index vectors pointing to bundles which
    //have that peel set size in the range of set sizes that the
    //playout bundle covers.
    let mut peel_set_size_index_map : Vec<Vec<usize>> = vec![Vec::new(); peel_set_size_spread];
    for i in 0..num_bundles {
        let bundle_state = bundle_states[i].as_ref().unwrap();
        let init_set_size = bundle_state.init_set_size;
        let final_set_size = bundle_state.final_set_size;
        //TODO: Maybe compact these ahead-of-time, so that they don't even enter into
        //the peel calculations section?
        //Special case: void all bundle states which take exactly one turn to solve
        //[and therefore, are already solved before getting to the peel]
        if (init_set_size + 1 == final_set_size) {
            bundle_states[i] = Option::None
        } else {
            //Note that we do not include the final set size, because that's what we're
            //going to be stepping _to_ [for the last move], and therefore that won't
            //enter into our training data.
            for set_size in (init_set_size + 1)..final_set_size {
                let set_size_offset = set_size - min_peel_set_size;
                peel_set_size_index_map[set_size_offset].push(i);
            }
        }
    }

    //Loop through possible set sizes for the peel
    for (set_size_offset, mut indices_for_size) in peel_set_size_index_map.drain(..)
                                                                    .enumerate() {
        //Pull out all of the bundle states which overlap this set size
        let mut bundle_states_for_size = Vec::new();
        for index in &indices_for_size {
            let bundle_state = std::mem::replace(&mut bundle_states[*index], Option::None).unwrap();
            bundle_states_for_size.push(bundle_state);
        }
        if (bundle_states_for_size.is_empty()) {
            continue;
        }
        //Take the bundle states and advance them
        let (partial_loss, mut updated_bundles) = BundleState::advance(network_config, params,
                                                                       bundle_states_for_size);

        //Contribute partial loss to the total
        total_loss += partial_loss;

        //Put the bundles back in their proper places, but dropping bundles
        //which have no more remaining moves
        for (index, updated_bundle) in indices_for_size.drain(..).zip(updated_bundles.drain(..)) {
            let maybe_bundle = 
                if updated_bundle.has_remaining_moves() {
                    Option::Some(updated_bundle)
                } else {
                    Option::None
                };
            bundle_states[index] = maybe_bundle;
        }
    }

    total_loss
}
