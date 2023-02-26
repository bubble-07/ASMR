use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::array_utils::*;
use rand::Rng;
use rand::seq::SliceRandom;
use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use fixedbitset::*;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct GamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub indegree : usize,
    pub cumulative_turns : FixedBitSet,
}

pub struct AnnotatedGamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub child_visit_probabilities : Array2<f32>,
}

/// A semantic path that a game can take, but not involving
/// anything relating to starting matrices, targets, visit
/// probabilities, etc.
#[derive(Clone)]
pub struct GamePath {
    pub initial_set_size : usize,
    pub nodes : Vec<GamePathNode>
}

pub struct AnnotatedGamePath {
    pub initial_set_size : usize,
    pub nodes : Vec<AnnotatedGamePathNode>
}

impl AnnotatedGamePath {
    pub fn get_num_turns(&self) -> usize {
        self.nodes.len()
    }
    pub fn get_initial_set_size(&self) -> usize {
        self.initial_set_size
    }
}

pub fn apply_orthonormal_basis_change(matrices : Tensor, Q : Tensor) -> Tensor {
    //Q.t().dot(matrices).dot(Q)
    unimplemented!();
}

/// N x M x M -> N x M x M
pub fn derive_orthonormal_basis_changes_from_target_matrices(target_matrices : &Tensor) -> Tensor {
    let _guard = no_grad_guard();
    let n = target_matrices.size()[0];
    let m = target_matrices.size()[1];
    //TODO: May want schur decomposition instead here,
    //or possibly a smarter algorithm which actually produces
    //the best dominant subspaces
    let transposed = target_matrices.transpose(1, 2);
    let symmetrized : Tensor = 0.5 * transposed + 0.5 * target_matrices;
    let antisymmetrized = target_matrices - &symmetrized;
    //Eigenvalues are default-sorted ascending
    //eigenvalues : N x M, eigenvectors : N x M x M - cols are eigenvectors
    let (eigenvalues, eigenvectors) = symmetrized.linalg_eigh("U");
    //Rows are eigenvectors
    let eigenvectors = eigenvectors.transpose(1, 2);

    //Find permutation matrices to sort eigenvalues in descending order of absolute value
    let abs_eigenvalues = eigenvalues.abs();
    //N x M, indices of sorted elements
    let sort_indices = abs_eigenvalues.argsort(1, true);
    //N x M x M, indices of sorted elements repeated so we can use a gather
    let sort_indices = sort_indices.unsqueeze(2);
    let sort_indices = sort_indices.expand(&[-1, -1, m], false);
    
    //Now, permute eigenvectors to match the abs eigenvalues' new ordering
    //N x M x M, rows are eigenvectors
    let eigenvectors = eigenvectors.gather(1, &sort_indices, false);
    let Q_T = eigenvectors.shallow_clone();
    let Q = eigenvectors.transpose(1, 2);

    //Finally, we need to resolve the * +-1 factor for each of the eigenvectors.
    //We do this by trial-transforming the antisymmetric part, and then ensuring
    //that all of the row-sums are positive
    //N x M x M
    let trial_transformed = Q_T.matmul(&antisymmetrized).matmul(&Q);
    let trial_sums = trial_transformed.sum_dim_intlist(Some(&[2 as i64] as &[i64]), false, Kind::Float);
    //N x M
    let trial_signs = trial_sums.sign();
    //Fix up the signs so that we map zeroes -> 1, so it's only -1/+1
    let trial_signs = (trial_signs + 0.5).sign();
    //Expand the trial signs to be the same size as the eigenvectors
    let trial_signs = trial_signs.unsqueeze(2);
    let trial_signs = trial_signs.expand(&[-1, -1, m], false);

    //Use the trial signs to determine the orientation
    let eigenvectors = trial_signs * eigenvectors;
    let Q_T = eigenvectors.shallow_clone();
    let Q = eigenvectors.transpose(1, 2);

    Q
}

impl GamePathNode {
    pub fn annotate(self, child_visit_probabilities : Array2<f32>) -> AnnotatedGamePathNode {
        AnnotatedGamePathNode {
            left_index : self.left_index,
            right_index : self.right_index,
            child_visit_probabilities,
        }
    }
}


impl GamePath {
    pub fn new(initial_set_size : usize) -> Self {
        Self {
            initial_set_size,
            nodes : Vec::new(),
        }
    }

    pub fn annotate_path(mut self) -> AnnotatedGamePath {
        let mut child_visit_probabilities_by_added_node = self.get_child_visit_probabilities_by_added_node();
        let mut annotated_nodes = Vec::new();

        for (node, child_visit_probabilities) in self.nodes.drain(..)
                                                 .zip(child_visit_probabilities_by_added_node.drain(..)) {
            let annotated_node = node.annotate(child_visit_probabilities);
            annotated_nodes.push(annotated_node);
        }

        AnnotatedGamePath {
            initial_set_size : self.initial_set_size,
            nodes : annotated_nodes
        }
    }

    pub fn get_child_visit_probabilities_by_added_node(&self) -> Vec<Array2<f32>> {
        let mut successors_per_node : Vec<HashSet<usize>> = self.get_potential_successors_by_added_node();
        let initial_set_size = self.get_initial_set_size();
        let mut result = Vec::new();

        for (mut successor_index_set, current_node_index) in successors_per_node.drain(..)
                                                             .zip(0..self.nodes.len()) {
            let k_plus_t = initial_set_size + current_node_index;
            let mut child_visit_probability_mat = Array::zeros((k_plus_t, k_plus_t));
            let increment : f32 = 1.0f32 / (successor_index_set.len() as f32);
            for index in successor_index_set.drain() {
                let node_index = self.to_node_index(index);
                let node = &self.nodes[node_index];
                let left_index = node.left_index;
                let right_index = node.right_index;
                child_visit_probability_mat[[left_index, right_index]] += increment;
            }
            result.push(child_visit_probability_mat);
        }
        result
    }

    //Returned mapping is from actual node index added to the entire set of possible nodes
    //which could have been added at that particular point in the game path
    pub fn get_potential_successors_by_added_node(&self) -> Vec<HashSet<usize>> {
        let mut result = Vec::new();

        let reverse_links = self.get_reverse_links();

        let init_set_size = self.get_initial_set_size();
        for current_index in init_set_size..(init_set_size + self.nodes.len()) {
            let mut potential_successors = HashSet::new();
            for contained_index in 0..current_index {
                let next_indices = &reverse_links[contained_index];
                for next_index in next_indices.iter() {
                    if (*next_index >= current_index) {
                        let next_node_index = *next_index - init_set_size;
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
        let init_set_size = self.get_initial_set_size();
        for i in 0..init_set_size {
            let mut reverse_links_for_current = HashSet::new();
            for j in 0..self.nodes.len() {
                let node = &self.nodes[j];
                if (node.left_index == i || node.right_index == i) {
                    let current_index = init_set_size + j;
                    reverse_links_for_current.insert(current_index);
                }
            }
            reverse_links.push(reverse_links_for_current);
        }
        //Add reverse links for nodes -> nodes
        for i in 0..self.nodes.len() {
            let origin_index = init_set_size + i;
            let mut reverse_links_for_current = HashSet::new();
            for j in i..self.nodes.len() {
                let node = &self.nodes[j];
                if (node.left_index == origin_index || node.right_index == origin_index) {
                    let current_index = init_set_size + j;
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

        let init_set_size = self.get_initial_set_size();

        //First, add all nodes [ignoring the very last one] which have indegree zero
        for i in 0..(self.nodes.len() - 1) {
            if (self.nodes[i].indegree == 0) {
                let index = init_set_size + i;
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
                let orig_index = init_set_size + i;
                if (removed_indices.contains(&orig_index)) {
                    removed_elements_so_far += 1;
                } else {
                    let replacement_index = orig_index - removed_elements_so_far;
                    index_renumberings.insert(orig_index, replacement_index);
                }
            }

            let mut updated_nodes = Vec::new();
            for (i, node) in self.nodes.drain(..).enumerate() {
                let orig_index = init_set_size + i;
                if !removed_indices.contains(&orig_index) {

                    let left_index = if node.left_index >= init_set_size {
                        *index_renumberings.get(&node.left_index).unwrap()
                    } else {
                        node.left_index
                    };

                    let right_index = if node.right_index >= init_set_size {
                        *index_renumberings.get(&node.right_index).unwrap()
                    } else {
                        node.right_index
                    };


                    let updated_node = GamePathNode {
                        left_index,
                        right_index,
                        indegree : node.indegree,
                        cumulative_turns : node.cumulative_turns, //TODO: Doesn't actually update this,
                        //despite the fact that indices have moved
                    };
                    updated_nodes.push(updated_node);
                }
            }
            self.nodes = updated_nodes;
        }
    }
    
    pub fn get_initial_set_size(&self) -> usize {
        self.initial_set_size
    }

    pub fn get_size(&self) -> usize {
        self.get_initial_set_size() + self.nodes.len()
    }

    fn is_node_index(&self, index : usize) -> bool {
        index >= self.get_initial_set_size()
    }

    fn to_node_index(&self, index : usize) -> usize {
        index - self.get_initial_set_size()
    }

    ///Adds a node with the given left, right indices.
    ///Returns the number of cumulative turns under the added node
    pub fn add_node(&mut self, left_index : usize, right_index : usize) -> usize {
        let indegree = 0;
        
        //Mark this turn's attendance
        let mut cumulative_turns = FixedBitSet::with_capacity(self.nodes.len() + 1);
        cumulative_turns.insert(self.nodes.len());
       
        if (self.is_node_index(left_index)) {
            let left_node_index = self.to_node_index(left_index);
            let left_node = &mut self.nodes[left_node_index];
            left_node.indegree += 1;
            //Union the cumulative turns
            cumulative_turns.union_with(&left_node.cumulative_turns);
        }
        if (self.is_node_index(right_index)) {
            let right_node_index = self.to_node_index(right_index);
            let right_node = &mut self.nodes[right_node_index];
            right_node.indegree += 1;
            cumulative_turns.union_with(&right_node.cumulative_turns);
        }

        let cumulative_num_turns = cumulative_turns.count_ones(..);
 
        let game_path_node = GamePathNode {
            left_index,
            right_index,
            indegree,
            cumulative_turns,
        };

        self.nodes.push(game_path_node);
        cumulative_num_turns
    }
    
    ///Returns the number of cumulative turns under the added node
    fn add_random_node<R : Rng + ?Sized>(&mut self, already_added_pairs : &mut HashSet<(usize, usize)>,
                                             rng : &mut R) -> usize {
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
        self.add_node(left_index, right_index)
    }
    
    pub fn generate_game_path<R : Rng + ?Sized>(initial_set_size : usize, 
                                                ground_truth_num_moves : usize,
                                                rng : &mut R) -> GamePath {
        //let iter_bound = ground_truth_num_moves;
        //TODO: There's probably a sensible bound to create?
        let iter_bound = initial_set_size + ground_truth_num_moves;
        let iter_bound = iter_bound * iter_bound;
        //Repeatedly try generating a game path
        loop {
            let mut already_added_pairs = HashSet::new();
            let mut result = GamePath::new(initial_set_size);
            //Generate a bunch of potential moves until one's cumulative number
            //of turns under it matches what we wanted to generate.
            for _ in 0..iter_bound {
                let cumulative_turns = result.add_random_node(&mut already_added_pairs, rng);
                if (cumulative_turns == ground_truth_num_moves) {
                    //GC unnecessary nodes in the generated game-path
                    result.garbage_collect();
                    return result;
                }
            }
        }
        //return result;
    }
}
