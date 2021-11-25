extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::network_config::*;
use crate::array_utils::*;
use crate::game_state::*;
use crate::normal_inverse_chi_squared::*;

use rand::Rng;

use tch::{kind, Tensor};

pub struct GameTreeNode {
    pub maybe_expanded_edges : Option<ExpandedEdges>,
    pub current_distance : f32,
    pub game_end_distance_distribution : NormalInverseChiSquared
}

pub struct ExpandedEdges {
    pub children_start_index : usize,
    pub edges : Vec<GameTreeEdge>
}

pub struct GameTreeEdge {
    pub added_matrix : Array2<f32>,
    pub visit_count : usize
}

pub struct GameTree {
    pub nodes : Vec<GameTreeNode>,
    pub init_game_state : GameState
}

pub struct GameTreeTraverser {
    pub index_stack : Vec<usize>,
    pub game_state : GameState
}

impl GameTree {
    pub fn traverse_from_root(&self) -> GameTreeTraverser {
        let mut index_stack = Vec::new();
        index_stack.push(0);

        let game_state = self.init_game_state.clone();
        GameTreeTraverser {
            index_stack,
            game_state
        }
    }
    pub fn update_iteration<R : Rng + ?Sized>(&mut self, network_config : &NetworkConfig, rng : &mut R) {
        let mut traverser = self.traverse_from_root();
        while (traverser.has_expanded_children(&self) && traverser.has_remaining_turns()) {
            traverser.move_to_best_child(self, rng);
        }
        if (!traverser.has_remaining_turns()) {
            //All-expanded children, but we ran out of turns. No big deal,
            //this traversal was still probably useful to update the edge visit-counts
            return;
        }
        //Otherwise, we must be at a place where we should expand all of the children.
        traverser.expand_children(self, network_config, rng);        
    }
}

impl GameTreeTraverser {
    fn current_node_index(&self) -> usize {
        self.index_stack[self.index_stack.len() - 1]
    }
    fn current_node<'a>(&self, game_tree : &'a GameTree) -> &'a GameTreeNode {
        &game_tree.nodes[self.current_node_index()]
    }
    pub fn has_expanded_children(&self, game_tree : &GameTree) -> bool {
        self.current_node(game_tree).maybe_expanded_edges.is_some()
    }

    pub fn has_remaining_turns(&self) -> bool {
        self.game_state.remaining_turns > 0
    }
    
    ///Assumes that the children have already been expanded
    fn move_to_best_child<R : Rng + ?Sized>(&mut self, game_tree : &mut GameTree, rng : &mut R) {
        let current_node_index = self.current_node_index();

        let (num_edges, children_start_index, normalized_distance_distribution) = {
            let current_node = &game_tree.nodes[current_node_index];

            let current_game_end_distance_distribution = current_node.game_end_distance_distribution;
            let normalized_distance_distribution = current_game_end_distance_distribution.as_single_observation();

            let expanded_edges = &current_node.maybe_expanded_edges.as_ref().unwrap();
            let edges = &expanded_edges.edges;

            let num_edges = edges.len();
            let children_start_index = expanded_edges.children_start_index;

            (num_edges, children_start_index, normalized_distance_distribution)
        };

        let mut min_index = 0;
        let mut min_value = f64::INFINITY;

        for i in 0..num_edges {
            let child_index = children_start_index + i;
            let child = &game_tree.nodes[child_index];

            let child_game_end_distance_distribution = child.game_end_distance_distribution;
            let combined_game_end_distance_distribution = child_game_end_distance_distribution.merge(&normalized_distance_distribution);

            let sampled_distance = combined_game_end_distance_distribution.sample(rng);
            if (sampled_distance < min_value) {
                min_value = sampled_distance;
                min_index = i;
            }
        }

        let current_node = &mut game_tree.nodes[current_node_index];
        let edges = &mut current_node.maybe_expanded_edges.as_mut().unwrap().edges;
        
        edges[min_index].visit_count += 1;

        let best_node_index = children_start_index + min_index;
        self.index_stack.push(best_node_index); 
    }

    fn expand_children<R : Rng + ?Sized>(&self, game_tree : &mut GameTree, 
                                         network_config : &NetworkConfig, rng : &mut R) {

        let parent_rollout_state = network_config.start_rollout(self.game_state.clone());

        let current_node_index = self.current_node_index();
        let current_node = &game_tree.nodes[current_node_index];

        let current_game_end_distance_distribution = current_node.game_end_distance_distribution;

        let current_set_size = self.game_state.get_num_matrices();
        let current_distance = self.game_state.get_distance();
        let target = self.game_state.get_target();

        let mut edges = Vec::new();

        let children_start_index = game_tree.nodes.len();
        for i in 0..current_set_size {
            let left_matrix = self.game_state.matrix_set.get(i);
            for j in 0..current_set_size {
                let right_matrix = self.game_state.matrix_set.get(j);

                let added_matrix = left_matrix.dot(&right_matrix);

                let dist_to_added = sq_frob_dist(added_matrix.view(), target);
                let child_current_distance = current_distance.min(dist_to_added);

                //We're going to create a roll-out for the node in a bit, which
                //is why the visit count here is 1 and not zero.
                let visit_count = 1;

                let edge = GameTreeEdge {
                    added_matrix : added_matrix.clone(),
                    visit_count
                };

                edges.push(edge);

                //Perform a rollout
                let mut child_rollout_state = parent_rollout_state.clone();
                child_rollout_state = network_config.manual_step_rollout(child_rollout_state, added_matrix);
                let rollout_distance = network_config.complete_rollout(child_rollout_state, rng);

                //Children get a single-rollout-updated prior.
                let mut game_end_distance_distribution = NormalInverseChiSquared::uninformative();
                game_end_distance_distribution = game_end_distance_distribution.update(rollout_distance as f64);

                let node = GameTreeNode {
                    maybe_expanded_edges : Option::None,
                    current_distance : child_current_distance,
                    game_end_distance_distribution
                };
                game_tree.nodes.push(node);
            }
        }
        let expanded_edges = ExpandedEdges {
            children_start_index,
            edges
        };
        game_tree.nodes[current_node_index].maybe_expanded_edges = Option::Some(expanded_edges);
    }
}
