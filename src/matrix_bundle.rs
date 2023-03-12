use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};
use crate::synthetic_data::*;
use std::collections::HashMap;
use crate::training_examples::*;
use std::ops::Range;
use std::iter::zip;

pub struct MatrixBundle {
    ///Tensor of dims NxKxM, where
    ///N is the number of example playouts, and M is the flattened matrix dimension
    pub flattened_initial_matrix_sets : Tensor,
    ///Dims NxM
    pub flattened_matrix_targets : Tensor,
}

fn remove(named_tensor_map : &mut HashMap<String, Tensor>, key : &str) -> Result<Tensor, String> {
    named_tensor_map.remove(key).ok_or(format!("Missing key {}", key))
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
    let (eigenvectors, eigenvalues, _) = Tensor::linalg_svd(&symmetrized, true, "gesvdj");

    //This is actually really slow due to a combined pytorch+MAGMA bug
    //let (eigenvalues, eigenvectors) = symmetrized.linalg_eigh("U");

    //Rows are eigenvectors after doing this, previously they were columns
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

    //Finally, we need to resolve the * +-1 factor for each of the eigenvectors (up to
    //an overall multiplication by +-I)
    //We do this by trial-transforming the target matrices, and then ensuring
    //that all of the elements in the first row are non-negative.
    //N x M x M
    let trial_transformed = Q_T.matmul(&target_matrices).matmul(&Q);
    //N x M
    let trial_transformed_first_row = trial_transformed.i((.., 0, ..));
    //N x M
    let trial_signs = trial_transformed_first_row.sign();
    //Fix up the signs so that we map zeroes -> 1, so it's only -1/+1
    let trial_signs = (trial_signs + 0.5).sign();
    //Expand the trial signs to be the same size as the eigenvectors
    let trial_signs = trial_signs.unsqueeze(2);
    let trial_signs = trial_signs.expand(&[-1, -1, m], false);

    //Use the trial signs to determine the orientation
    let eigenvectors = trial_signs * eigenvectors;
    let Q = eigenvectors.transpose(1, 2);

    Q
}

/// Applies the orthonormal basis transforms given in Q to the given matrices,
/// all of which are assumed to be N x M x M
fn apply_orthonormal_basis_transform(Q : &Tensor, matrices : &Tensor) -> Tensor {
    let Q_T = Q.transpose(1, 2); 
    Q_T.matmul(matrices).matmul(Q)
}

impl MatrixBundle {
    pub fn to_device(&self, device : Device) -> Self {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.to_device(device);
        let flattened_matrix_targets = self.flattened_matrix_targets.to_device(device);
        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        }
    }
    /// Takes a collection of orthonormal basis-change matrices [dims N x M_sqrt x M_sqrt]
    /// and applies the basis changes to each collection of matrices in this matrix bundle
    pub fn apply_orthonormal_basis_transforms(&self, Q : &Tensor) -> Self {
        let n = self.get_num_playouts() as i64;
        let k = self.get_init_set_size() as i64;
        let m = self.get_flattened_matrix_dim() as i64;
        let m_sqrt = (m as f64).sqrt() as i64;

        let reshaped_matrix_targets = self.flattened_matrix_targets.reshape(&[n, m_sqrt, m_sqrt]);
        let transformed_matrix_targets = apply_orthonormal_basis_transform(Q, &reshaped_matrix_targets);

        let Q_T = Q.transpose(1, 2); 
        let Q = Q.reshape(&[n, 1, m_sqrt, m_sqrt]);
        let Q_T = Q_T.reshape(&[n, 1, m_sqrt, m_sqrt]);

        let reshaped_initial_matrix_sets = self.flattened_initial_matrix_sets.reshape(&[n, k, m_sqrt, m_sqrt]);
        
        let transformed_initial_matrix_sets = Q_T.matmul(&reshaped_initial_matrix_sets).matmul(&Q);

        let flattened_initial_matrix_sets = transformed_initial_matrix_sets.reshape(&[n, k, m]);
        let flattened_matrix_targets = transformed_matrix_targets.reshape(&[n, m]);

        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        }
    }
    pub fn standardize(&self) -> Self {
        let n = self.get_num_playouts() as i64;
        let m = self.get_flattened_matrix_dim() as i64;
        let m_sqrt = (m as f64).sqrt() as i64;
        let reshaped_matrix_targets = self.flattened_matrix_targets.reshape(&[n, m_sqrt, m_sqrt]);
        //N x M_sqrt x M_sqrt
        let Q = derive_orthonormal_basis_changes_from_target_matrices(&reshaped_matrix_targets);
        self.apply_orthonormal_basis_transforms(&Q)
    }
    pub fn device(&self) -> Device {
        self.flattened_matrix_targets.device()
    }
    pub fn get_init_set_size(&self) -> usize {
        self.flattened_initial_matrix_sets.size()[1] as usize
    }
    pub fn get_flattened_matrix_dim(&self) -> usize {
        self.flattened_matrix_targets.size()[1] as usize
    }
    fn concat_consume(a : Tensor, b : Tensor) -> Tensor {
        let result = Tensor::cat(&[a, b], 0);
        result
    }
    pub fn shallow_clone(&self) -> Self {
        Self {
            flattened_initial_matrix_sets : self.flattened_initial_matrix_sets.shallow_clone(),
            flattened_matrix_targets : self.flattened_matrix_targets.shallow_clone(),
        }
    }
    pub fn merge_all(bundles : Vec<Self>) -> Self {
        let flattened_initial_matrix_sets : Vec<_> = bundles.iter().map(|x| x.flattened_initial_matrix_sets.shallow_clone()).collect();
        let flattened_matrix_targets : Vec<_> = bundles.iter().map(|x| x.flattened_matrix_targets.shallow_clone()).collect();

        let flattened_initial_matrix_sets = Tensor::concat(&flattened_initial_matrix_sets, 0);
        let flattened_matrix_targets = Tensor::concat(&flattened_matrix_targets, 0);

        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        }
    }
    pub fn split(self, split_sizes : &[i64]) -> Vec<Self> {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.split_with_sizes(split_sizes, 0);
        let flattened_matrix_targets = self.flattened_matrix_targets.split_with_sizes(split_sizes, 0);
        zip(flattened_initial_matrix_sets, flattened_matrix_targets)
        .map(|(flattened_initial_matrix_sets, flattened_matrix_targets)|
             Self {
                flattened_initial_matrix_sets,
                flattened_matrix_targets,
            })
        .collect()
    }
}

impl PlayoutBundleLike for MatrixBundle {
    fn get_num_playouts(&self) -> usize {
        self.flattened_matrix_targets.size()[0] as usize
    }
    fn grab_batch(&self, batch_index_range : Range<i64>, device : Device) -> MatrixBundle {
        let flattened_initial_matrix_sets = self.flattened_initial_matrix_sets.i(batch_index_range.clone())
                                            .to_device(device).detach();

        let flattened_matrix_targets = self.flattened_matrix_targets.i(batch_index_range.clone())
                                       .to_device(device).detach();
        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        }
    }
    fn merge(mut self, mut other : Self) -> Self {
        let _guard = no_grad_guard();
        let flattened_initial_matrix_sets = Self::concat_consume(self.flattened_initial_matrix_sets,
                                                                  other.flattened_initial_matrix_sets);
        let flattened_matrix_targets = Self::concat_consume(self.flattened_matrix_targets,
                                                            other.flattened_matrix_targets);
        Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        }
    }
    fn serialize(mut self, prefix : String) -> Vec<(String, Tensor)> {
        let mut result = Vec::new();

        result.push((format!("{}_initial_matrix_sets", prefix),
                     self.flattened_initial_matrix_sets));

        result.push((format!("{}_flattened_matrix_targets", prefix),
                    self.flattened_matrix_targets));

        result
    }

    fn load(named_tensor_map : &mut HashMap<String, Tensor>, key : (usize, usize))
           -> Result<Self, String> {
        let (initial_set_size, playout_length) = key;
        let prefix = format!("{}_{}", initial_set_size, playout_length);

        let flattened_initial_matrix_sets = remove(named_tensor_map,
                                            &format!("{}_initial_matrix_sets", prefix))?;

        let flattened_matrix_targets = remove(named_tensor_map,
                                       &format!("{}_flattened_matrix_targets", prefix))?;
        Result::Ok(Self {
            flattened_initial_matrix_sets,
            flattened_matrix_targets,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate;
    #[test]
    fn test_orthonormal_basis_invariance_of_standardization() {
        let N = 10;
        let M = 10;
        let log_normal_std_dev = 1.0f64;
        let device = tch::Device::Cuda(0);
        
        //Test that if we randomly rotate the basis for some random matrices,
        //standardizing yields the same result as standardizing the original random matrices
        let random_matrices = generate::random_log_normal_singular_value_matrices(N, M, log_normal_std_dev, device);

        let standardizing_basis_changes = derive_orthonormal_basis_changes_from_target_matrices(&random_matrices);
        let standardized_matrices = apply_orthonormal_basis_transform(&standardizing_basis_changes, &random_matrices);
        println!("standardized matrices: {}", standardized_matrices.to_string(80).unwrap());
        

        let random_basis_changes = generate::random_orthogonal_matrices(N, M, device);
        let randomized_matrices = apply_orthonormal_basis_transform(&random_basis_changes, &random_matrices);
        
        let restandardizing_basis_changes = derive_orthonormal_basis_changes_from_target_matrices(&randomized_matrices);
        let restandardized_matrices = apply_orthonormal_basis_transform(&restandardizing_basis_changes, &randomized_matrices);

        println!("restandardized matrices: {}", restandardized_matrices.to_string(80).unwrap());
        
        //Now, the restandardized matrices should be identical to the standardized ones
        let diff = standardized_matrices - restandardized_matrices;
        let diff_squared = &diff * &diff;
        let total_diff_squared = f32::from(diff_squared.sum(Kind::Float));

        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.01) {
            panic!("Got different results for standard form after orthonormal basis transform!");
        }
    }
}

