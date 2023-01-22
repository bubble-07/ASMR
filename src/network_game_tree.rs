use std::rc::Rc;
use rand::Rng;
use crate::game_tree_trait::*;
use crate::network_config::*;
use crate::tree::*;
use crate::game_state::*;
use crate::rollout_states::*;
use crate::network_rollout::*;
use crate::normal_inverse_chi_squared::*;

pub struct NetworkTreeBase {
    //Rollout states with exactly one rollout,
    //and just starting from the base. 
    pub network_rollout_state : NetworkRolloutState,
    pub network_config : Rc<NetworkConfig>,
    pub ordinary_root_data : OrdinaryRootData,
}

pub struct NetworkTreeNode {
    pub ordinary_data : OrdinaryNodeData,
}

pub struct NetworkTreeEdge {
    ///Optionally, we cache the diff in the network rollout state.
    ///Intended to only really be done for super-high-traffic edges
    pub precomputed_data : Option<NetworkRolloutDiff>,
    pub ordinary_data : OrdinaryEdgeData,
}

pub struct NetworkTreeTraverserData {
    pub network_rollout_state : NetworkRolloutState,
}

type NetworkTreeTraverserType = TreeWithTraverser<NetworkTreeTraverserData,
                                                       NetworkTreeBase, NetworkTreeNode, NetworkTreeEdge>;

pub struct NetworkTreeTraverser {
    pub tree_traverser : NetworkTreeTraverserType,
}

impl HasTreeWithTraverser for NetworkTreeTraverser {
    type TraverserState = NetworkTreeTraverserData;
    type BaseData = NetworkTreeBase;
    type NodeData = NetworkTreeNode;
    type EdgeData = NetworkTreeEdge;
    fn get(&self) -> &NetworkTreeTraverserType {
        &self.tree_traverser
    }
    fn get_mut(&mut self) -> &mut NetworkTreeTraverserType {
        &mut self.tree_traverser
    }
}

impl NetworkTreeTraverser {
    fn get_network_rollout_states(&self) -> &NetworkRolloutState {
        &self.tree_traverser.get_traverser_state().network_rollout_state
    }
    pub fn build_from_game_state(network_config : Rc<NetworkConfig>, game_state : GameState) -> Self {
        let rollout_state = RolloutStates::from_single_game_state(&game_state); 
        let network_rollout_state = NetworkRolloutState::from_rollout_states(&network_config, rollout_state);

        let network_tree_traverser_data = NetworkTreeTraverserData {
            network_rollout_state : network_rollout_state.shallow_clone(),
        };

        let ordinary_root_node_data = OrdinaryNodeData::root_node_from_game_state(&game_state);
        let root_node_data = NetworkTreeNode {
            ordinary_data : ordinary_root_node_data,
        };

        let ordinary_root_data = OrdinaryRootData::from_single_game_state(game_state);
        let root_data = NetworkTreeBase {
            ordinary_root_data,
            network_config,
            network_rollout_state,
        };

        let tree_traverser = TreeWithTraverser::new(root_data, root_node_data,
                                                    network_tree_traverser_data);
        Self {
            tree_traverser
        }
    }
}

impl GameTreeTraverserTrait for NetworkTreeTraverser {
    fn drain_indices_to_root(&mut self) -> Vec<NodeIndex> {
        let network_rollout_state = self.tree_traverser.get_root_data().network_rollout_state.shallow_clone();
        let traverser_state = NetworkTreeTraverserData {
            network_rollout_state,
        };
        self.tree_traverser.drain_indices_to_root(traverser_state)
    }
    fn get_rollout_states(&self) -> &RolloutStates {
        &self.get_network_rollout_states().rollout_states
    }
    fn get_ordinary_root_data(&self) -> &OrdinaryRootData {
        &self.tree_traverser.get_root_data().ordinary_root_data
    }
    fn get_ordinary_edge_data(&self, child_edge_index : EdgeIndex) -> &OrdinaryEdgeData {
        &self.tree_traverser.get_edge_data(child_edge_index).ordinary_data
    }
    fn get_ordinary_edge_data_mut(&mut self, child_edge_index : EdgeIndex) -> &mut OrdinaryEdgeData {
        &mut self.tree_traverser.get_edge_data_mut(child_edge_index).ordinary_data
    }
    fn get_ordinary_node_data_for(&mut self, node_index : NodeIndex) -> &mut OrdinaryNodeData {
        &mut self.tree_traverser.get_tree_mut().get_node_data_for(node_index).ordinary_data
    }
    fn move_to_child(&mut self, child_edge_index : EdgeIndex) {

        let edge_data = self.get_ordinary_edge_data_mut(child_edge_index);
        edge_data.visit_count += 1;
        let left_index = edge_data.left_index;
        let right_index = edge_data.right_index;

        let network_rollout_state = self.get_network_rollout_states().shallow_clone();

        let precomputed_data = &self.tree_traverser.get_edge_data(child_edge_index).precomputed_data;
        let network_rollout_state = match precomputed_data {
            Some(network_rollout_diff) => {
                network_rollout_state.apply_diff(network_rollout_diff)
            },
            None => {
                let tree = self.tree_traverser.get_tree();
                let network_config = &tree.base_data.network_config;

                network_rollout_state.perform_move(network_config, left_index, right_index)
            },
        };

        let traverser_state = NetworkTreeTraverserData {
            network_rollout_state,
        };
        self.tree_traverser.manual_move(child_edge_index, traverser_state);
    }
    fn expand_children(&mut self, rng : &mut DynRng) -> Vec<f32> {
        let tree = self.tree_traverser.get_tree();
        let network_config = &tree.base_data.network_config;
        //First, expand the current single-rollout rollout state to incorporate
        //all possibilities for the children
        let network_rollout_states = self.get_network_rollout_states().shallow_clone();
        let current_set_size = network_rollout_states.rollout_states.get_num_matrices() as usize;

        let child_rollout_states = network_rollout_states.perform_all_moves(network_config);
        //Determine how many turns are left, and also the current set-size
        let child_num_turns = child_rollout_states.rollout_states.remaining_turns;
        let current_distances : Vec<f64> = child_rollout_states.rollout_states.min_distances.shallow_clone().into();

        //If there are child turns left, finish 'em out with random rollouts
        let child_rollout_states = child_rollout_states.complete_network_rollouts(network_config);
        let mut ending_distances : Vec<f64> = child_rollout_states.rollout_states.min_distances.shallow_clone().into();

        //Edge data, Node data
        let mut child_tuples = Vec::new();

        //Fill in the edge + node data
        for left_index in 0..current_set_size {
            for right_index in 0..current_set_size {
                let combined_index = left_index + current_set_size * right_index;
                let edge_data = NetworkTreeEdge {
                    ordinary_data : OrdinaryEdgeData {
                        left_index,
                        right_index,
                        visit_count : 1,
                    },
                    precomputed_data : None, //TODO: Store precomputed data in these nodes
                };
                let current_distance = current_distances[combined_index];
                let ending_distance = ending_distances[combined_index];

                let game_end_distance_distribution = if child_num_turns > 0 {
                    NormalInverseChiSquared::Uninformative.update(ending_distance) 
                } else {
                    NormalInverseChiSquared::Certain(current_distance)
                };
                let current_distance = current_distance as f32;
                let node_data = NetworkTreeNode {
                    ordinary_data : OrdinaryNodeData {
                        current_distance,
                        game_end_distance_distribution,
                    },
                };
                child_tuples.push((edge_data, node_data));
            }
        }
        self.tree_traverser.add_children(child_tuples);

        let ending_distances = ending_distances.drain(..).map(|x| x as f32).collect();
        ending_distances
    }
}
