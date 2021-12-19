use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::matrix_set::*;
use crate::game_data::*;
use crate::array_utils::*;
use rand::Rng;
use rand::seq::SliceRandom;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct GamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub indegree : usize
}

pub struct GamePath {
    pub matrix_set : MatrixSet,
    pub nodes : Vec<GamePathNode>
}

impl GamePath {
    pub fn new(matrix_set : MatrixSet) -> GamePath {
        GamePath {
            matrix_set,
            nodes : Vec::new()
        }
    }

    pub fn get_game_data(&self) -> GameData {
        let added_matrices = self.get_added_matrices();
        let mut successors_per_node : Vec<HashSet<usize>> = self.get_potential_successors_by_added_node();
        
        let mut flattened_matrix_sets = Vec::new();
        let mut child_visit_probabilities = Vec::new();

        for (mut successor_index_set, current_node_index) in successors_per_node.drain(..)
                                                             .zip(0..self.nodes.len()) {
            let current_index = self.matrix_set.size() + current_node_index;
            let mut flattened_matrices = Vec::new();
            for index in 0..current_index {
                let flattened_matrix = flatten_matrix(added_matrices.get(index)).to_owned();
                flattened_matrices.push(flattened_matrix);
            }

            let mut child_visit_probability_mat = Array::zeros((flattened_matrices.len(), flattened_matrices.len()));
            let increment : f32 = 1.0f32 / (successor_index_set.len() as f32);
            for index in successor_index_set.drain() {
                let node_index = self.to_node_index(index);
                let node = &self.nodes[node_index];
                let left_index = node.left_index;
                let right_index = node.right_index;
                child_visit_probability_mat[[left_index, right_index]] += increment;
            }

            flattened_matrix_sets.push(flattened_matrices);
            child_visit_probabilities.push(child_visit_probability_mat);
        }

        let flattened_matrix_target = flatten_matrix(added_matrices.get_newest_matrix()).to_owned();
    
        GameData {
            flattened_matrix_sets,
            flattened_matrix_target,
            child_visit_probabilities
        }
    }

    //Returned mapping is from actual node index added to the entire set of possible nodes
    //which could have been added at that particular point in the game path
    pub fn get_potential_successors_by_added_node(&self) -> Vec<HashSet<usize>> {
        let mut result = Vec::new();

        let reverse_links = self.get_reverse_links();

        for current_index in self.matrix_set.size()..(self.matrix_set.size() + self.nodes.len()) {
            let mut potential_successors = HashSet::new();
            for contained_index in 0..current_index {
                let next_indices = &reverse_links[contained_index];
                for next_index in next_indices.iter() {
                    if (*next_index >= current_index) {
                        let next_node_index = *next_index - self.matrix_set.size(); 
                        let next_node = &self.nodes[next_node_index];
                        if (next_node.left_index < current_index &&
                            next_node.right_index < current_index) {
                            potential_successors.insert(*next_index);
                        }
                    }
                }
            }
            result.push(potential_successors);
        }
        result
    }
 
    fn get_reverse_links(&self) -> Vec<HashSet<usize>> {
        let mut reverse_links = Vec::new();
        //Add reverse links for original set -> nodes
        for i in 0..self.matrix_set.size() {
            let mut reverse_links_for_current = HashSet::new();
            for j in 0..self.nodes.len() {
                let node = &self.nodes[j];
                if (node.left_index == i || node.right_index == i) {
                    let current_index = self.matrix_set.size() + j;
                    reverse_links_for_current.insert(current_index);
                }
            }
            reverse_links.push(reverse_links_for_current);
        }
        //Add reverse links for nodes -> nodes
        for i in 0..self.nodes.len() {
            let origin_index = self.matrix_set.size() + i;
            let mut reverse_links_for_current = HashSet::new();
            for j in i..self.nodes.len() {
                let node = &self.nodes[j];
                if (node.left_index == origin_index || node.right_index == origin_index) {
                    let current_index = self.matrix_set.size() + j;
                    reverse_links_for_current.insert(current_index);
                }
            }
            reverse_links.push(reverse_links_for_current);
        }
        reverse_links
    }
                                        

    pub fn garbage_collect(&mut self) {

        let mut indegree_zero_indices = Vec::new();
        let mut removed_indices = HashSet::new();

        //First, add all nodes [ignoring the very last one] which have indegree zero
        for i in 0..(self.nodes.len() - 1) {
            if (self.nodes[i].indegree == 0) {
                let index = i + self.matrix_set.size();
                indegree_zero_indices.push(index);
            }
        }

        //Then, repeatedly remove nodes which reach indegree zero from removals
        while (indegree_zero_indices.len() > 0) {
            let removed_index = indegree_zero_indices.pop().unwrap();
            removed_indices.insert(removed_index);

            let removed_node_index = self.to_node_index(removed_index);
            let removed_node = &self.nodes[removed_node_index];
            let left_index = removed_node.left_index;
            let right_index = removed_node.right_index;

            if (self.is_node_index(left_index)) {
                let left_node_index = self.to_node_index(left_index);
                self.nodes[left_node_index].indegree -= 1;
                if (self.nodes[left_node_index].indegree == 0) {
                    indegree_zero_indices.push(left_index);
                }
            }

            if (self.is_node_index(right_index)) {
                let right_node_index = self.to_node_index(right_index);
                self.nodes[right_node_index].indegree -= 1;
                if (self.nodes[right_node_index].indegree == 0) {
                    indegree_zero_indices.push(right_index);
                }
            } 
        }
        
        //Finally, compact the node-set according to the removed indices
        if (removed_indices.len() > 0) {
            let mut index_renumberings = HashMap::new();

            let mut removed_elements_so_far = 0;
            for i in 0..self.nodes.len() {
                let orig_index = i + self.matrix_set.size();
                if (removed_indices.contains(&orig_index)) {
                    removed_elements_so_far += 1;
                } else {
                    let replacement_index = orig_index - removed_elements_so_far;
                    index_renumberings.insert(orig_index, replacement_index);
                }
            }

            let mut updated_nodes = Vec::new();
            for node in self.nodes.drain(..) {
                let updated_node = GamePathNode {
                    left_index : *index_renumberings.get(&node.left_index).unwrap(),
                    right_index : *index_renumberings.get(&node.right_index).unwrap(),
                    indegree : node.indegree
                };
                updated_nodes.push(updated_node);
            }
            self.nodes = updated_nodes;
        }
    }

    fn get_added_matrices(&self) -> MatrixSet {
        let mut result = self.matrix_set.clone();
        for node in self.nodes.iter() {
            let added_matrix = {
                let left_matrix = result.get(node.left_index);
                let right_matrix = result.get(node.right_index);
                left_matrix.dot(&right_matrix)
            };
            result = result.add_matrix(added_matrix);
        }
        result
    }

    pub fn get_target(&self) -> Array2<f32> {
        let added_matrices = self.get_added_matrices();
        added_matrices.get_newest_matrix().to_owned()
    }

    pub fn get_size(&self) -> usize {
        self.matrix_set.size() + self.nodes.len()
    }

    fn is_node_index(&self, index : usize) -> bool {
        index >= self.matrix_set.size()
    }

    fn to_node_index(&self, index : usize) -> usize {
        index - self.matrix_set.size()
    }

    pub fn add_node(&mut self, left_index : usize, right_index : usize) {
        let indegree = 0;
        
        let game_path_node = GamePathNode {
            left_index,
            right_index,
            indegree
        };

        if (self.is_node_index(left_index)) {
            let left_node_index = self.to_node_index(left_index);
            self.nodes[left_node_index].indegree += 1;
        }
        if (self.is_node_index(right_index)) {
            let right_node_index = self.to_node_index(right_index);
            self.nodes[right_node_index].indegree += 1;
        }

        self.nodes.push(game_path_node);
    }
    
    fn add_random_node<R : Rng + ?Sized>(&mut self, already_added_pairs : &mut HashSet<(usize, usize)>,
                                             rng : &mut R) {
        let size = self.get_size();
        let mut left_index = rng.gen_range(0..size);
        let mut right_index = rng.gen_range(0..size);
        let mut pair = (left_index, right_index);
        while (already_added_pairs.contains(&pair)) {
            left_index = rng.gen_range(0..size);
            right_index = rng.gen_range(0..size);
            pair = (left_index, right_index);
        }
        already_added_pairs.insert(pair);
        self.add_node(left_index, right_index);
    }
    
    pub fn generate_game_path<R : Rng + ?Sized>(matrix_set : MatrixSet, 
                                                ground_truth_num_moves : usize,
                                                rng : &mut R) -> GamePath {
        let mut already_added_pairs = HashSet::new();

        let mut result = GamePath::new(matrix_set);
        for _ in 0..ground_truth_num_moves {
            result.add_random_node(&mut already_added_pairs, rng);
        }
        result
    }
}
