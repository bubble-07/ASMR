use ndarray::*;
use std::collections::HashSet;
use std::collections::HashMap;
use crate::array_utils::*;
use rand::Rng;
use rand::seq::SliceRandom;
use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use fixedbitset::*;

#[derive(Clone)]
pub struct AnnotatedGamePathNode {
    pub left_index : usize,
    pub right_index : usize,
    pub child_visit_probabilities : Array2<f32>,
}

/// A semantic path that the game can take, but not
/// involving anything relating to starting matrices
/// or targets, just visit probabilities and actually-chosen
/// matrices.
pub struct AnnotatedGamePath {
    pub initial_set_size : usize,
    pub nodes : Vec<AnnotatedGamePathNode>
}

/// Structure representing a string of matrices
/// that are multiplied together [from left to right]
#[derive(Clone)]
pub struct MultipliedMatrices {
    pub indices: Vec<usize>,
}

impl MultipliedMatrices {
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn get_indices(&self) -> &[usize] {
        self.indices.as_slice()
    }

    /// Generates a random `MultipliedMatrices` with the given
    /// initial set-size and the given string length (indices.len).
    pub fn random_with_init_set_size_and_string_length<R : Rng + ?Sized>(
        initial_set_size : usize,
        index_string_length : usize,
        rng : &mut R
    ) -> Self {
        let mut indices = Vec::new();
        for _ in 0..index_string_length {
            let index = rng.gen_range(0..initial_set_size);
            indices.push(index);
        }
        Self { indices }
    }

    /// Gets a list of all adjacent index-pairs in this `MultipliedMatrices`
    pub fn get_adjacent_index_pairs(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for i in 0..(self.indices.len() - 1) {
            result.push((self.indices[i], self.indices[i + 1])); 
        }
        result
    }

    /// Replaces all instances of the given adjacent index-pair
    /// in this with the given single index
    pub fn replace_index_pair(&self, pattern_index_pair: (usize, usize), replacement: usize) -> Self {
        let mut result = Vec::new();
        let mut i = 0;
        while i < (self.indices.len() - 1) {
            let current_index_pair = (self.indices[i], self.indices[i + 1]);
            if (current_index_pair == pattern_index_pair) {
                result.push(replacement);
                i += 2;
            } else {
                // Push the left of the two indices
                result.push(self.indices[i]);
                i += 1;
            }
        }
        //If we make it to the last index, include that one, too.
        //We may have skipped over this, depending on the route taken
        //in the loop.
        if (i == self.indices.len() - 1) {
            result.push(self.indices[i]);
        }
        Self { indices: result }
    }
}

impl AnnotatedGamePath {
    pub fn get_num_turns(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_initial_set_size(&self) -> usize {
        self.initial_set_size
    }

    pub fn get_size(&self) -> usize {
        self.get_initial_set_size() + self.nodes.len()
    }
       
    pub fn generate_game_path<R : Rng + ?Sized>(initial_set_size : usize, 
                                                ground_truth_num_moves : usize,
                                                rng : &mut R) -> Self {
        let mut nodes = Vec::new();

        //TODO: I know that the "ground_truth_num_moves" here isn't quite respected
        //in the case of having duplicate adjacent pairs, but it doesn't really
        //matter for now.
        let mut multiplied_matrices = MultipliedMatrices::random_with_init_set_size_and_string_length(
            initial_set_size, ground_truth_num_moves + 1, rng
        );
        //This will track the current number of matrices in the final set of the
        //game-path that is being generated.
        let mut current_set_size = initial_set_size;

        // When our multiplied matrix string is reduced to a single matrix, we've reached the
        // target.
        while multiplied_matrices.len() > 1 {
            // Get all pairs of indices that are adjacent in the matrix-multiply string.
            let adjacent_index_list = multiplied_matrices.get_adjacent_index_pairs();

            // Derive child visit probabilities from the sensible multiplications of
            // adjacent matrices in the current matrix string.
            let child_visit_probabilities = index_pairs_to_visit_matrix(current_set_size, 
                                                                        &adjacent_index_list);

            // Pick a privileged pair of adjacent indices for the next move
            let adjacent_indices = adjacent_index_list[rng.gen_range(0..adjacent_index_list.len())];

            // Replace the privileged pair with a freshly-generated matrix index
            multiplied_matrices = multiplied_matrices.replace_index_pair(adjacent_indices, current_set_size);

            // Just added a new matrix, so increment the counter.
            current_set_size += 1;

            let (left_index, right_index) = adjacent_indices;

            let annotated_game_path_node = AnnotatedGamePathNode {
                left_index,
                right_index,
                child_visit_probabilities
            };

            nodes.push(annotated_game_path_node);
        }

        Self {
            initial_set_size,
            nodes,
        }
    }
}

fn index_pairs_to_visit_matrix(set_size : usize, index_pairs : &[(usize, usize)]) -> Array2<f32> {
    let mut child_visit_probability_mat = Array::zeros((set_size, set_size));
    let increment : f32 = 1.0f32 / (index_pairs.len() as f32);
    for (left_index, right_index) in index_pairs {
        child_visit_probability_mat[[*left_index, *right_index]] += increment;
    }
    child_visit_probability_mat
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_adjacent_index_pairs() {
        let two_element_test = MultipliedMatrices {
            indices: vec![13, 20],
        };
        let expected_index_pairs = vec![(13, 20)];
        let index_pairs = two_element_test.get_adjacent_index_pairs();        
        assert_eq!(expected_index_pairs.as_slice(), index_pairs);

        let five_element_test = MultipliedMatrices {
            indices: vec![9, 4, 55, 100, 29],
        };
        let expected_index_pairs = vec![(9, 4), (4, 55), (55, 100), (100, 29)];
        let index_pairs = five_element_test.get_adjacent_index_pairs();        
        assert_eq!(expected_index_pairs.as_slice(), index_pairs);
    }

    #[test]
    fn test_replace_single_index_pair() {
        let multiplied_matrices = MultipliedMatrices {
            indices: vec![99, 11, 33, 7],
        };
        let updated_matrices = multiplied_matrices.replace_index_pair((33, 7), 6);
        assert_eq!(updated_matrices.get_indices(), &[99, 11, 6]);

        let updated_matrices = multiplied_matrices.replace_index_pair((11, 33), 2);
        assert_eq!(updated_matrices.get_indices(), &[99, 2, 7]);

        let updated_matrices = multiplied_matrices.replace_index_pair((99, 11), 4);
        assert_eq!(updated_matrices.get_indices(), &[4, 33, 7]);
    }
}
