use rand::Rng;
use crate::game_tree_trait::*;
use crate::tree::*;
use crate::rollout_states::*;
use crate::network_config::*;
use crate::normal_inverse_chi_squared::*;

pub struct RandomTreeBase {
    pub ordinary_root_data : OrdinaryRootData,
}

pub struct RandomTreeNode {
    pub data : OrdinaryNodeData,
}

pub struct RandomTreeEdge {
    pub data : OrdinaryEdgeData,
}

pub struct RandomTreeTraverserData {
    pub rollout_state : RolloutStates,
}

type RandomTreeTraverserType = TreeWithTraverser<RandomTreeTraverserData,
                                                      RandomTreeBase, RandomTreeNode, RandomTreeEdge>;

pub struct RandomTreeTraverser {
    pub tree_traverser : RandomTreeTraverserType,
}

impl HasTreeWithTraverser for RandomTreeTraverser {
    type TraverserState = RandomTreeTraverserData;
    type BaseData = RandomTreeBase;
    type NodeData = RandomTreeNode;
    type EdgeData = RandomTreeEdge;
    fn get(&self) -> &RandomTreeTraverserType {
        &self.tree_traverser
    }
    fn get_mut(&mut self) -> &mut RandomTreeTraverserType {
        &mut self.tree_traverser
    }
}

impl RandomTreeTraverser {
    pub fn build_from_game_state(rollout_state : RolloutStates) -> Self {
        let random_tree_traverser_data = RandomTreeTraverserData {
            rollout_state : rollout_state.shallow_clone(),
        };

        let ordinary_root_node_data = OrdinaryNodeData::root_node_from_game_state(&rollout_state);
        let root_node_data = RandomTreeNode {
            data : ordinary_root_node_data,
        };

        let ordinary_root_data = OrdinaryRootData::from_single_game_state(rollout_state);
        let root_data = RandomTreeBase {
            ordinary_root_data,
        };

        let tree_traverser = TreeWithTraverser::new(root_data, root_node_data,
                                                    random_tree_traverser_data);
        Self {
            tree_traverser
        }
    }
}

impl GameTreeTraverserTrait for RandomTreeTraverser {
    fn drain_indices_to_root(&mut self) -> Vec<NodeIndex> {
        let rollout_state = self.tree_traverser.get_root_data().ordinary_root_data.game_state.shallow_clone();
        let traverser_state = RandomTreeTraverserData {
            rollout_state,
        };
        self.tree_traverser.drain_indices_to_root(traverser_state)
    }
    fn get_rollout_states(&self) -> &RolloutStates {
        &self.tree_traverser.get_traverser_state().rollout_state
    }
    fn get_ordinary_root_data(&self) -> &OrdinaryRootData {
        &self.tree_traverser.get_root_data().ordinary_root_data
    }
    fn get_ordinary_edge_data(&self, child_edge_index : EdgeIndex) -> &OrdinaryEdgeData {
        &self.tree_traverser.get_edge_data(child_edge_index).data
    }
    fn get_ordinary_edge_data_mut(&mut self, child_edge_index : EdgeIndex) -> &mut OrdinaryEdgeData {
        &mut self.tree_traverser.get_edge_data_mut(child_edge_index).data
    }
    fn get_ordinary_node_data_for(&mut self, node_index : NodeIndex) -> &mut OrdinaryNodeData {
        &mut self.tree_traverser.get_tree_mut().get_node_data_for(node_index).data
    }
    fn move_to_child(&mut self, child_edge_index : EdgeIndex) {
        let edge_data = self.get_ordinary_edge_data_mut(child_edge_index);
        edge_data.visit_count += 1;
        let left_index = edge_data.left_index;
        let right_index = edge_data.right_index;

        let rollout_state = self.tree_traverser.get_traverser_state().rollout_state.shallow_clone();
        let rollout_state = rollout_state.perform_move(left_index, right_index);
        let traverser_state = RandomTreeTraverserData {
            rollout_state,
        };
        self.tree_traverser.manual_move(child_edge_index, traverser_state);
    }
    fn expand_children(&mut self, rng : &mut DynRng) -> Vec<f32> {
        //First, expand the current single-rollout rollout state to incorporate
        //all possibilities for the children
        let rollout_states = self.get_rollout_states().shallow_clone();
        let current_set_size = rollout_states.get_num_matrices() as usize;

        let child_rollout_states = rollout_states.perform_all_moves();
        //Determine how many turns are left, and also the current set-size
        let child_num_turns = child_rollout_states.remaining_turns;
        let current_distances : Vec<f64> = child_rollout_states.min_distances.shallow_clone().into();

        //If there are child turns left, finish 'em out with random rollouts
        let child_rollout_states = child_rollout_states.complete_random_rollouts();
        let mut ending_distances : Vec<f64> = child_rollout_states.min_distances.shallow_clone().into();

        //Edge data, Node data
        let mut child_tuples = Vec::new();

        //Fill in the edge + node data
        for left_index in 0..current_set_size {
            for right_index in 0..current_set_size {
                let combined_index = left_index + current_set_size * right_index;
                let edge_data = RandomTreeEdge {
                    data : OrdinaryEdgeData {
                        left_index,
                        right_index,
                        visit_count : 1,
                    },
                };
                let current_distance = current_distances[combined_index];
                let ending_distance = ending_distances[combined_index];

                let game_end_distance_distribution = if child_num_turns > 0 {
                    NormalInverseChiSquared::Uninformative.update(ending_distance) 
                } else {
                    NormalInverseChiSquared::Certain(current_distance)
                };
                let current_distance = current_distance as f32;
                let node_data = RandomTreeNode {
                    data : OrdinaryNodeData {
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

//pub fn new(init_game_state : GameState) -> GameTree {
//        let current_distance = init_game_state.distance;
//
 //       //Let the initial game-state just have an observation of the initial distance.
//        //We don't really care what our initialization looks like, tbh, so long as it provides
//        //a valid prior distribution to propagate down the tree.
//        let game_end_distance_distribution = NormalInverseChiSquared::Uninformative.update(current_distance as f64);
//        
//        let init_node_data = GameTreeNode {
//            current_distance,
//            game_end_distance_distribution,
//        };
//        GameTree {
//            tree : Tree::new(init_game_state, init_node_data),
//        }
//    }
