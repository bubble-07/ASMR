use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};

use std::fmt;
use crate::array_utils::*;

pub struct MatrixSets {
    ///The matrices in each set, each of dims MxM, R x (k+t) of 'em
    ///R x (k + t) x M x M
    matrices : Tensor,
}

pub struct MatrixSetDiff {
    ///The matrices which were added at this step
    ///R x 1 x M x M
    matrices : Tensor,
}

impl fmt::Display for MatrixSets {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.matrices.to_string(80).unwrap())
    }
}

impl fmt::Display for MatrixSetDiff {
    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "added: {}", self.matrices.to_string(80).unwrap())
    }
}

impl MatrixSetDiff {
    pub fn matrices(&self) -> &Tensor {
        &self.matrices
    }
    pub fn split_to_singles(self) -> Vec<Self> {
        let matrices = self.matrices.split(1, 0);
        matrices.into_iter()
                .map(|matrices| Self {
                    matrices
                })
                .collect()
    }
    pub fn get_flattened_added_matrices(&self) -> Tensor {
        let r = self.matrices.size()[0];
        let m = self.matrices.size()[2];
        self.matrices.reshape(&[r, m * m])
    }
}

impl MatrixSets {
    pub fn new(matrices : Tensor) -> Self {
        Self {
            matrices,
        }
    }
    pub fn get_flattened_matrices(&self) -> Tensor {
        let r = self.matrices.size()[0];
        let k = self.matrices.size()[1];
        let m = self.matrices.size()[2];
        self.matrices.reshape(&[r, k, m * m])
    }
    pub fn matrices(&self) -> &Tensor {
        &self.matrices
    }
    pub fn device(&self) -> Device {
        self.matrices.device()
    }
    pub fn shallow_clone(&self) -> Self {
        Self {
            matrices : self.matrices.shallow_clone(),
        }
    }
    pub fn apply_diff(self, diff : &MatrixSetDiff) -> Self {
        let matrices = Tensor::concat(&[self.matrices, diff.matrices.shallow_clone()], 1);
        Self {
            matrices,
        }
    }
    pub fn merge(sets : Vec<Self>) -> Self {
        let matrices : Vec<Tensor> = sets.into_iter().map(|x| x.matrices.shallow_clone()).collect();
        let matrices = Tensor::concat(&matrices, 0);
        Self {
            matrices,
        }
    }

    pub fn manual_step(self, left_indices : &Tensor, right_indices : &Tensor) -> Self {
        let diff = self.perform_moves_diff(left_indices, right_indices);
        self.apply_diff(&diff)
    }

    pub fn split(self, split_sizes : &[i64]) -> Vec<Self> {
        let matrices = self.matrices.split_with_sizes(split_sizes, 0);
        matrices.into_iter()
                .map(|matrices| Self {
                    matrices,
                })
                .collect()
    }
    pub fn get_num_sets(&self) -> i64 {
        self.matrices.size()[0]
    }
    pub fn get_matrix_size(&self) -> i64 {
        self.matrices.size()[2]
    }
    pub fn get_num_matrices(&self) -> i64 {
        self.matrices.size()[1]
    }

    //Given N x L tensors for the left and right indices, respectively, for a sequence of moves,
    //returns the flattened target matrices (N x M) obtained at the end of the sequence of moves
    pub fn get_flattened_targets_from_moves(&self, left_matrix_indices : &Tensor, 
                                                  right_matrix_indices : &Tensor) -> Tensor {
        let mut zelf = self.shallow_clone();
    
        let mut left_matrix_indices = left_matrix_indices.unbind(1);
        left_matrix_indices.reverse();

        let mut right_matrix_indices = right_matrix_indices.unbind(1);
        right_matrix_indices.reverse();
        //Roll forward the target-finding rollout 
        let playout_length = left_matrix_indices.len();
        for _ in 0..(playout_length - 1) {
            let left_indices = left_matrix_indices.pop().unwrap();
            let right_indices = right_matrix_indices.pop().unwrap();
            zelf = zelf.manual_step(&left_indices, &right_indices);
        }
        //Perform the final step, for which we'll only need the diff
        let left_indices = left_matrix_indices.pop().unwrap();
        let right_indices = right_matrix_indices.pop().unwrap();
        let final_step_diff = zelf.perform_moves_diff(&left_indices, &right_indices);

        final_step_diff.get_flattened_added_matrices()
    }

    pub fn perform_moves_diff(&self, left_indices : &Tensor, right_indices : &Tensor) -> MatrixSetDiff {
        let _guard = no_grad_guard();
        let r = self.get_num_sets();
        let m = self.get_matrix_size();
    

        let left_indices = left_indices.reshape(&[r, 1, 1, 1]);
        let right_indices = right_indices.reshape(&[r, 1, 1, 1]);

        let expanded_shape = vec![r, 1, m, m];

        let left_indices = left_indices.expand(&expanded_shape, false);
        let right_indices = right_indices.expand(&expanded_shape, false);

        //Gets the left/right matrices which were sampled for the next step in rollouts
        //Dimensions R x M x M
        let left_matrices = self.matrices.gather(1, &left_indices, false);
        let right_matrices = self.matrices.gather(1, &right_indices, false);

        let left_matrices = left_matrices.reshape(&[r, m, m]);
        let right_matrices = right_matrices.reshape(&[r, m, m]);

        //R x M x M
        let matrices = left_matrices.matmul(&right_matrices);
        let matrices = matrices.reshape(&[r, 1, m, m]);
        MatrixSetDiff {
            matrices,
        }
    }
    pub fn expand(self, R : usize) -> Self {
        let _guard = no_grad_guard();
        let matrices = self.matrices.expand(&[R as i64, -1, -1, -1], false); 
        Self {
            matrices,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_targets_from_moves() {
        let matrices_shape = vec![2, 4, 3, 3];
        let num_elements = 2 * 4 * 3 * 3;

        let matrices_flat : Tensor = 0.1f32 * Tensor::arange(num_elements, (Kind::Float, Device::Cpu));
        let matrices = matrices_flat.reshape(&matrices_shape);

        println!("Matrices: {}", matrices.to_string(80).unwrap());

        let matrices = MatrixSets::new(matrices);

        let left_indices = Tensor::from( &[0 as i64, 4, 5, 1, 3, 2] as &[i64]).reshape(&[2, 3]);
        let right_indices = Tensor::from(&[2 as i64, 1, 0, 0, 2, 4] as &[i64]).reshape(&[2, 3]);

        println!("Left indices: {}", left_indices.to_string(80).unwrap());
        println!("Right indices: {}", right_indices.to_string(80).unwrap());

        let flattened_targets = matrices.get_flattened_targets_from_moves(&left_indices, &right_indices);
        println!("Flattened targets: {}", flattened_targets.to_string(80).unwrap());

        //Expected output for the playouts
        let flattened_expected_output : Vec<f32> = 
            vec![2.673, 3.5208, 4.3686, 10.0278, 13.2084, 16.389, 17.3826, 22.896, 28.4094,
                 947.6370, 971.9100, 996.1830, 999.2881, 1024.8840, 1050.4801, 1050.9390, 1077.8580,
                 1104.7771];
        let flattened_expected_output = Tensor::from(&flattened_expected_output as &[f32]).reshape(&[2, 9]);
        
        let total_diff_squared = tensor_diff_squared(&flattened_targets, &flattened_expected_output);
        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.01) {
            panic!("Got different results for finding target matrix!");
        }
    }
}
