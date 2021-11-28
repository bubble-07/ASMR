extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;
use ndarray_linalg::*;

use crate::network_config::*;
use crate::array_utils::*;
use crate::game_state::*;
use crate::normal_inverse_chi_squared::*;
use crate::game_data::*;
use std::fmt;

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

#[derive(Clone)]
pub struct GameTreeTraverser {
    pub index_stack : Vec<usize>,
    pub game_state : GameState
}


impl GameTree {
    pub fn new(init_game_state : GameState) -> GameTree {
        let mut nodes = Vec::new();

        let current_distance = init_game_state.distance;

        //Let the initial game-state just have an observation of the initial distance.
        //We don't really care what our initialization looks like, tbh, so long as it provides
        //a valid prior distribution to propagate down the tree.
        let game_end_distance_distribution = NormalInverseChiSquared::Uninformative.update(current_distance as f64);

        let init_node = GameTreeNode {
            maybe_expanded_edges : Option::None,
            current_distance,
            game_end_distance_distribution
        };
        nodes.push(init_node);
        GameTree {
            nodes,
            init_game_state
        }
    }

    pub fn render_dotfile(&self) -> String {
        let traverser = self.traverse_from_root();
        let content = self.render_dotfile_recursive(traverser);
        format!("digraph gametree {{\n {} \n }}", content)
    }

    fn render_dotfile_recursive(&self, traverser : GameTreeTraverser) -> String {
        let current_node_index = traverser.current_node_index();

        let label_str = if (current_node_index == 0) {
            format!("{}", self.init_game_state)
        } else {
            let node = traverser.current_node(self);
            format!("{}", node)
        };

        let mut result = format!("{} [label=\"{}\"];\n", current_node_index, label_str);

        let mut child_tuples = traverser.get_child_tuples(&self);
        if (child_tuples.len() > 0) {
            for (child_index, added_matrix, visit_count) in child_tuples.drain(..) {
                let label = format!("added: {}, visits: {}", added_matrix, visit_count);
                result += &format!("{} -> {} [label=\"{}\"];\n", current_node_index, child_index, label);

                let child_traverser = traverser.clone().manual_move(child_index, added_matrix);
                result += &self.render_dotfile_recursive(child_traverser);
            }
        }
        result
    }

    pub fn extract_game_data(&self) -> GameData {
        let flattened_matrix_target = flatten_matrix(self.init_game_state.target.view()).to_owned();

        let mut result = GameData {
            flattened_matrix_sets : Vec::new(),
            flattened_matrix_target,
            child_visit_probabilities : Vec::new()
        };

        let traverser = self.traverse_from_root();

        self.extract_game_data_recursive(&mut result, traverser);
        result
    }

    fn extract_game_data_recursive(&self, result : &mut GameData,
                                                             traverser : GameTreeTraverser) {
        let num_children = traverser.get_num_children(&self);
        let num_children_sqrt = (num_children as f64).sqrt() as usize;
        //We only need to do something here if we have children
        if (num_children > 0) {
            let mut child_tuples = traverser.get_child_tuples(&self);
            let mut child_visit_count_matrix = Array::zeros((num_children_sqrt, num_children_sqrt));
            let mut total_visits = 0;
            for (child_tuple, i) in child_tuples.drain(..).zip(0..num_children) {
                let (child_index, added_matrix, visit_count) = child_tuple;

                let child_traverser = traverser.clone().manual_move(child_index, added_matrix);
                self.extract_game_data_recursive(result, child_traverser);
                
                let left_index = i / num_children_sqrt;
                let right_index = i % num_children_sqrt;

                total_visits += visit_count;
                child_visit_count_matrix[(left_index, right_index)] = visit_count;
            }
            let child_visit_probability_matrix = child_visit_count_matrix
                                                 .mapv(|x| ((x as f64) / (total_visits as f64)) as f32);
            
            let flattened_matrix_set = traverser.game_state.matrix_set.get_flattened_vectors().iter()
                                                          .map(|x| x.to_owned()).collect();


            result.flattened_matrix_sets.push(flattened_matrix_set);
            result.child_visit_probabilities.push(child_visit_probability_matrix);
        }
    }

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
        let child_values = traverser.expand_children(self, network_config, rng);        
        traverser.update_distributions_to_root(self, child_values);
    }
}

impl fmt::Display for GameTreeNode {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "current distance: {} \n game end distance: {}", 
               self.current_distance, &self.game_end_distance_distribution)
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
    pub fn get_remaining_turns(&self) -> usize {
        self.game_state.get_remaining_turns()
    }

    pub fn update_distributions_to_root(mut self, game_tree : &mut GameTree, mut child_values : Vec<f32>) {
        let mut update_distribution = NormalInverseChiSquared::Uninformative;
        for child_value in child_values.drain(..) {
            update_distribution = update_distribution.update(child_value as f64);
        }
        for parent_node_index in self.index_stack.drain(..).rev() {
            let current_game_end_distance_distribution = 
                game_tree.nodes[parent_node_index].game_end_distance_distribution;

            game_tree.nodes[parent_node_index].game_end_distance_distribution = 
                current_game_end_distance_distribution.merge(&update_distribution);
        }
    }

    pub fn manual_move(self, child_index : usize, added_matrix : Array2<f32>) -> Self {
        let mut index_stack = self.index_stack;
        index_stack.push(child_index);

        let game_state = self.game_state.add_matrix(added_matrix);

        GameTreeTraverser {
            index_stack,
            game_state
        }
    }

    ///Format is (child index, added matrix, visit count)
    pub fn get_child_tuples(&self, game_tree : &GameTree) -> Vec<(usize, Array2<f32>, usize)> {
        let maybe_expanded_edges = self.current_node(game_tree).maybe_expanded_edges.as_ref();
        match (maybe_expanded_edges) {
            Option::Some(expanded_edges) => {
                let mut result = Vec::new();

                let children_start_index = expanded_edges.children_start_index;
                let edges = &expanded_edges.edges;                
                for i in 0..edges.len() {
                    let edge = &edges[i];
                    let child_index = children_start_index + i;
                    let tuple = (child_index, edge.added_matrix.clone(), edge.visit_count);
                    result.push(tuple);
                }
                result
            },
            Option::None => Vec::new()
        }
    }

    pub fn get_num_children(&self, game_tree : &GameTree) -> usize {
        let maybe_expanded_edges = self.current_node(game_tree).maybe_expanded_edges.as_ref();
        match (maybe_expanded_edges) {
            Option::Some(expanded_edges) => {
                expanded_edges.edges.len()
            },
            Option::None => 0
        }
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
            //Has to be non-degenerate by construction, since it's a parent node
            let nondegenerate = current_game_end_distance_distribution.coerce_to_nondegenerate().unwrap();
            let single_observation = nondegenerate.as_single_observation();
            let normalized_distance_distribution = NormalInverseChiSquared::NonDegenerate(single_observation);

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
        let added_matrix = edges[min_index].added_matrix.clone();

        let best_node_index = children_start_index + min_index;
        self.index_stack.push(best_node_index); 
        self.game_state = self.game_state.clone().add_matrix(added_matrix);
    }

    ///Expands the children under the traverser's position using the given
    ///NetworkConfig for performing rollouts (if needed). Returns a collection
    ///of all of the distanced derived from rollouts (or from the actual values, if terminal nodes)
    ///which came from the expanded children
    fn expand_children<R : Rng + ?Sized>(&self, game_tree : &mut GameTree, 
                                         network_config : &NetworkConfig, rng : &mut R) -> Vec<f32> {

        let parent_rollout_state = network_config.start_rollout(self.game_state.clone());

        let current_node_index = self.current_node_index();

        let current_set_size = self.game_state.get_num_matrices();
        let current_distance = self.game_state.get_distance();
        let target = self.game_state.get_target();

        let mut result = Vec::new();

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

                //Prepare to perform a rollout if needed
                let mut child_rollout_state = parent_rollout_state.clone();
                child_rollout_state = network_config.manual_step_rollout(child_rollout_state, added_matrix);

                let child_num_turns = child_rollout_state.game_state.get_remaining_turns();

                let game_end_distance_distribution = if (child_num_turns > 0) {
                    //For a non-terminal node, we assign a single-rollout-updated uninformative prior.
                    let rollout_distance = network_config.complete_rollout(child_rollout_state, rng);
                    result.push(rollout_distance);
                    NormalInverseChiSquared::Uninformative.update(rollout_distance as f64)
                } else {
                    //For a terminal node, we assign the actual distance at the leaf
                    result.push(child_current_distance);
                    NormalInverseChiSquared::Certain(child_current_distance as f64)
                };

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
        result
    }
}
