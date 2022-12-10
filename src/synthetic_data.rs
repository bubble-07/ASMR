use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::matrix_set::*;
use crate::array_utils::*;
use rand::Rng;
use rand::seq::SliceRandom;
extern crate ndarray_linalg;
use ndarray_linalg::svd::*;
use ndarray_linalg::Norm;

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct GamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub indegree : usize
}

pub struct AnnotatedGamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub child_visit_probabilities : Array2<f32>,
    pub added_matrix : Array2<f32>
}

pub struct GamePath {
    pub matrix_set : MatrixSet,
    pub nodes : Vec<GamePathNode>
}

pub struct AnnotatedGamePath {
    pub matrix_set : MatrixSet,
    pub target_matrix : Array2<f32>,
    pub nodes : Vec<AnnotatedGamePathNode>
}

impl AnnotatedGamePathNode {
    pub fn apply_orthonormal_basis_change(self, Q : ArrayView2<f32>) -> Self {
        let added_matrix = Q.t().dot(&self.added_matrix).dot(&Q);
        Self {
            added_matrix,
            ..self
        }
    }
}

struct OrthonormalVectorSet(pub Vec<Array1<f32>>);

impl OrthonormalVectorSet {
    pub fn add_to_span(&mut self, mut to_add : Array1<f32>) {
        let min_length = 1e-2;

        let init_norm = to_add.norm_l2();
        if (init_norm < min_length) {
            //Not going to be a useful vector
            return;
        }
        //Normalize to_add
        to_add *= 1.0f32 / init_norm;

        //Orthogonalize
        for existing_vector in &self.0 {
            //Subtract off projection of to_add onto the existing
            let dot_product = to_add.dot(existing_vector);
            let projection = dot_product * existing_vector;
            to_add -= &projection;
        }
        //Make sure that the vector we're adding has a nonzero norm.
        //Otherwise, we'll skip adding it, since it's nearly linearly
        //dependent with the existing set of vectors.
        let to_add_length = to_add.norm_l2();
        if (to_add_length > min_length) {
            to_add *= 1.0f32 / to_add_length;
            //Presumably orthonormal now, throw it on
            self.0.push(to_add);
        }
    }
    pub fn to_orthonormal_matrix(self) -> Array2<f32> {
        let views : Vec<ArrayView1<f32>> = self.0.iter()
                          .map(|x| x.view())
                          .collect();
        stack(Axis(0), &views).unwrap()
    }
}

impl AnnotatedGamePath {
    pub fn derive_orthonormal_basis_change(&self) -> Array2<f32> {
        let (u, sigma, vt) = self.target_matrix.svd(true, true).unwrap();
        let u = u.unwrap();
        let vt = vt.unwrap();
        let ut = u.t();
        //Now, with U^T and V^T, the rows are the left and right
        //singular vectors, respectively. What we'll do is
        //start from the pairs of singular vectors corresponding
        //to the largest singular values, and add orthogonal vectors
        //spanning the subspaces spanned by each left and right singular vector
        let mut orthonormal_vector_set = OrthonormalVectorSet(Vec::new());
        for i in 0..ut.shape()[0] {
            let u_vec = ut.row(i);
            let v_vec = vt.row(i);

            let avg_vec = 0.5f32 * &u_vec + 0.5f32 * &v_vec;
            //Directionality is from input toward output
            let diff_vec = &u_vec - &v_vec;

            let dot_product = u_vec.dot(&v_vec);

            if (dot_product > 0.0) {
                //Same directionality, so the average will capture
                //the vectors better than the difference
                orthonormal_vector_set.add_to_span(avg_vec);
                orthonormal_vector_set.add_to_span(diff_vec);
            } else {
                //Opposite directionality, so the difference will
                //capture the vectors better than their average
                orthonormal_vector_set.add_to_span(diff_vec);
                orthonormal_vector_set.add_to_span(avg_vec);
            }
        }
        //Rows are in decreasing order of "importance", roughly.
        let orthonormal_matrix = orthonormal_vector_set.to_orthonormal_matrix();

        let (rows, cols) = (orthonormal_matrix.shape()[0], orthonormal_matrix.shape()[1]);
        if (rows != cols) {
            println!("Whoa there bucko {}, {}", rows, cols);
        }

        //Transpose the result, since we want to transform from
        //the shifted coordinate space back to the original one
        let orthonormal_matrix = orthonormal_matrix.t().clone();
        orthonormal_matrix.to_owned()
        
    }
    pub fn apply_orthonormal_basis_change(mut self, Q : ArrayView2<f32>) -> Self {
        let matrix_set = self.matrix_set.apply_orthonormal_basis_change(Q);
        let target_matrix = Q.t().dot(&self.target_matrix).dot(&Q);
        let nodes = self.nodes.drain(..)
                              .map(|x| x.apply_orthonormal_basis_change(Q))
                              .collect();
        AnnotatedGamePath {
            matrix_set,
            target_matrix,
            nodes
        }
    }
}

impl GamePathNode {
    pub fn annotate(self, child_visit_probabilities : Array2<f32>,
                               added_matrix : Array2<f32>) -> AnnotatedGamePathNode {
        AnnotatedGamePathNode {
            left_index : self.left_index,
            right_index : self.right_index,
            child_visit_probabilities,
            added_matrix
        }
    }
}


impl GamePath {
    pub fn new(matrix_set : MatrixSet) -> GamePath {
        GamePath {
            matrix_set,
            nodes : Vec::new()
        }
    }

    pub fn annotate_path(mut self) -> AnnotatedGamePath {
        let MatrixSet(added_matrices) = self.get_added_matrices();
        //Lop off the beginning, we don't care
        let mut added_matrices = added_matrices[self.get_initial_set_size()..].to_vec();

        let mut child_visit_probabilities_by_added_node = self.get_child_visit_probabilities_by_added_node();
        let mut annotated_nodes = Vec::new();
        
        for ((node, child_visit_probabilities), added_matrix) in self.nodes.drain(..)
                                                 .zip(child_visit_probabilities_by_added_node.drain(..))
                                                 .zip(added_matrices.drain(..)) {
            let annotated_node = node.annotate(child_visit_probabilities, added_matrix);
            annotated_nodes.push(annotated_node);
        }
        let target_matrix = annotated_nodes[annotated_nodes.len() - 1].added_matrix.clone();

        AnnotatedGamePath {
            matrix_set : self.matrix_set,
            target_matrix,
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

    pub fn get_initial_set_size(&self) -> usize {
        let MatrixSet(matrices) = &self.matrix_set;
        matrices.len()
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
