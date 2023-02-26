use rand::Rng;
use rand::RngCore;
use crate::tree::*;
use crate::network_config::*;
use crate::rollout_states::*;
use crate::normal_inverse_chi_squared::*;
use serde::{Serialize, Deserialize};

pub struct OrdinaryRootData {
    pub game_state : RolloutStates,
}

#[derive(Serialize, Deserialize)]
pub struct OrdinaryEdgeData {
    pub left_index : usize,
    pub right_index : usize,
    pub visit_count : usize,
}

impl OrdinaryRootData {
    pub fn from_single_game_state(game_state : RolloutStates) -> Self {
        OrdinaryRootData {
            game_state,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct OrdinaryNodeData {
    pub current_distance : f32,
    pub game_end_distance_distribution : NormalInverseChiSquared,
}

impl OrdinaryNodeData {
    pub fn root_node_from_game_state(game_state : &RolloutStates) -> Self {
        let current_distance = f32::from(&game_state.min_distances);
        let game_end_distance_distribution = NormalInverseChiSquared::Uninformative.update(current_distance as f64);
        OrdinaryNodeData {
            current_distance,
            game_end_distance_distribution,
        }
    }
}

pub struct DynRng<'a> {
    rng : &'a mut dyn RngCore,
}

impl <'a, R : RngCore> From<&'a mut R> for DynRng<'a> {
    fn from(rng : &'a mut R) -> Self {
        Self {
            rng : rng as &'a mut dyn RngCore,
        }
    }
}

impl <'a> RngCore for DynRng<'a> {
    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }
    fn fill_bytes(&mut self, dest : &mut [u8]) {
        self.rng.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest : &mut [u8]) -> Result<(), rand::Error> {
        self.rng.try_fill_bytes(dest)
    }
}

pub trait GameTreeTraverserTrait : TraverserLike {
    fn get_ordinary_root_data(&self) -> &OrdinaryRootData;
    fn get_rollout_states(&self) -> &RolloutStates;
    //Moves to the given child, updating the visit count
    fn move_to_child(&mut self, child_edge_index : EdgeIndex);
    fn get_ordinary_edge_data(&self, child_edge_index : EdgeIndex) -> &OrdinaryEdgeData;
    fn get_ordinary_edge_data_mut(&mut self, child_edge_index : EdgeIndex) -> &mut OrdinaryEdgeData;
    fn drain_indices_to_root(&mut self) -> Vec<NodeIndex>;

    fn get_ordinary_node_data_for(&mut self, node_index : NodeIndex) -> &mut OrdinaryNodeData;

    fn expand_children(&mut self, rng : &mut DynRng) -> Vec<f32>;

    fn get_ordinary_node_data(&mut self) -> &mut OrdinaryNodeData {
        let current_node_index = self.get_current_node_index();
        self.get_ordinary_node_data_for(current_node_index)
    }
    fn has_remaining_turns(&self) -> bool {
        self.get_rollout_states().remaining_turns > 0
    }
    ///Moves to the predicted-best child of the current node [based
    ///on Thompson sampling of the expected distance distributions].
    ///Assumes that the children have already been expanded.
    fn move_to_best_child(&mut self, rng : &mut DynRng) {
        //Treat the mean of the parent distribution as a single value,
        //with the intent of ensuring that every sampled child has at least two
        //data-points [and hence, a defined variance for sampling]
        let normalized_parent_distance_distribution = {
            let parent_game_end_distance_distribution = self.get_ordinary_node_data().game_end_distance_distribution;
            //Has to be non-degenerate by construction, since it's a parent node
            let nondegenerate = parent_game_end_distance_distribution.coerce_to_nondegenerate().unwrap();
            let single_observation = nondegenerate.as_single_observation();
            NormalInverseChiSquared::NonDegenerate(single_observation)
        };

        //Sample and find which child attains the minimum
        let mut min_index = None;
        let mut min_value = f64::INFINITY;

        for child_edge_index in self.get_child_edge_indices() {
            let child_node_index = child_edge_index.get_ending_node_index();
            let child = self.get_ordinary_node_data_for(child_node_index);

            let child_game_end_distance_distribution = child.game_end_distance_distribution;
            let combined_game_end_distance_distribution = child_game_end_distance_distribution.merge(
                &normalized_parent_distance_distribution);

            let sampled_distance = combined_game_end_distance_distribution.sample(rng);
            if (sampled_distance < min_value) {
                min_value = sampled_distance;
                min_index = Some(child_edge_index);
            }
        }
        self.move_to_child(min_index.unwrap());
    }

    ///For the given values of children lower in the tree, updates all currently-maintained
    ///parent distributions with these new observations. At the end of this, the traverser
    ///will be back at the root of the tree.
    fn update_distributions_to_root(&mut self, mut child_values : Vec<f32>) {
        let mut update_distribution = NormalInverseChiSquared::Uninformative;
        for child_value in child_values.drain(..) {
            update_distribution = update_distribution.update(child_value as f64);
        }
        let mut parent_indices : Vec<_> = self.drain_indices_to_root();
        for parent_index in parent_indices.drain(..) {
            let parent_node = self.get_ordinary_node_data_for(parent_index);

            let current_game_end_distance_distribution = parent_node.game_end_distance_distribution;

            parent_node.game_end_distance_distribution = 
                current_game_end_distance_distribution.merge(&update_distribution);
        }
    }

    ///Updates this game tree using MCTS with playouts determined by the passed network config
    ///and with random choices determined by the passed Rng. Returns the minimal distance to
    ///the target along the paths taken in the MCTS iteration, which includes the minimal
    ///distances achieved by the playouts during node-expansion.
    fn perform_update_iteration(&mut self, rng : &mut DynRng) -> f32 {
        //TODO: Probably want playing a game to use this only on the second run.
        //First run should just do a rollout straight from the root, and use that for
        //a preliminary answer. Then, the first call to "update_iteration" should
        //use that initial rollout to provide additional updated data? Maybe.
        while (self.has_expanded_children() && self.has_remaining_turns()) {
            self.move_to_best_child(rng);
        }
        if (!self.has_remaining_turns()) {
            //All-expanded children, but we ran out of turns. No big deal,
            //this traversal was still probably useful to update the edge visit-counts
            let distance = self.get_ordinary_node_data().current_distance;
            return distance;
        }
        //Otherwise, we must be at a place where we should expand all of the children.
        let child_values = self.expand_children(rng);        
        let mut minimal_child_value = f32::INFINITY; 
        for child_value in child_values.iter() {
            if (*child_value < minimal_child_value) {
                minimal_child_value = *child_value;
            }
        }
        self.update_distributions_to_root(child_values);

        minimal_child_value
    }
    
    ///Needs to be invoked with the traverser at the root
    fn render_dotfile(&mut self) -> String {
        let game_state = self.get_ordinary_root_data().game_state.shallow_clone();
        let content = self.render_dotfile_recursive(game_state);
        format!("digraph gametree {{\n {} \n }}", content)
    }

    fn render_dotfile_recursive(&mut self, game_state : RolloutStates) -> String {
        let current_node_index = self.get_current_node_index();

        let label_str = if (current_node_index == NodeIndex::from(0)) {
            format!("{}", game_state)
        } else {
            let node = self.get_ordinary_node_data();
            format!("distance: {}", node.current_distance)
        };

        let mut result = format!("{} [label=\"{}\"];\n", current_node_index, label_str);

        for child_edge_index in self.get_child_edge_indices() {
            let child_node_index = child_edge_index.get_ending_node_index();
            let edge_data = self.get_ordinary_edge_data(child_edge_index);

            let updated_game_state = game_state.shallow_clone();
            let updated_game_state_diff = game_state.perform_move_diff(edge_data.left_index, edge_data.right_index);
            let updated_game_state = updated_game_state.apply_diff(&updated_game_state_diff);
            
            let label = format!("{}, visits: {}", &updated_game_state_diff, edge_data.visit_count);
            result += &format!("{} -> {} [label=\"{}\"];\n", current_node_index, child_node_index, label);

            //Recursively render
            result += &self.render_dotfile_recursive(updated_game_state);

            //Done with that child, go back up
            self.go_to_parent_keep_state();
        }
        result
    }
}
