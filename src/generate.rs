use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};

/// Dims N X M X M
pub fn random_orthogonal_matrices(N : usize, M : usize, device : Device) -> Tensor {
    let N = N as i64;
    let M = M as i64;
    let normal_matrix = Tensor::randn(&[N, M, M], (Kind::Float, device));
    //Both N x M x M
    let (Q, R) = Tensor::linalg_qr(&normal_matrix, "reduced");

    //Correct non-uniqueness of decomposition by requiring R's diagonal
    //elements to be positive
    //N x M
    let D = R.diagonal(0, -2, -1);
    let signs = D.sign();
    //N x M x M
    let signs = signs.unsqueeze(1);
    let signs = signs.expand(&[-1, M, -1], true);
    let Q = Q * signs;
    
    Q
}

pub fn random_log_normal(log_normal_std_dev : f64, dims : &[i64], device : Device) -> Tensor {
    let normal_values = Tensor::randn(dims, (Kind::Float, device));
    let scaled_normal_values = log_normal_std_dev * normal_values;
    scaled_normal_values.exp()
}

pub fn random_log_normal_singular_value_matrices(N : usize, M : usize, log_normal_std_dev : f64, device : Device) -> Tensor {
    let U = random_orthogonal_matrices(N, M, device);
    let V = random_orthogonal_matrices(N, M, device);

    let N = N as i64;
    let M = M as i64;
    let S = random_log_normal(log_normal_std_dev, &[N, M], device);

    //Dims N x M x M
    let S = S.diag_embed(0, -2, -1);
    U.matmul(&S).matmul(&V) 
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_utils::*;
    #[test]
    fn test_matrices_orthonormal() {
        let N = 10;
        let M = 10;
        let device = tch::Device::Cuda(0);
        let random_matrices = random_orthogonal_matrices(N, M, device);
        let random_matrices_transposed = random_matrices.transpose(1, 2);
        let product = random_matrices.matmul(&random_matrices_transposed);

        let identities = Tensor::eye(M as i64, (Kind::Float, device));
        let identities = identities.unsqueeze(0);
        let total_diff_squared = tensor_diff_squared(&identities, &product);

        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.00001) {
            panic!("Generated matrices are not orthonormal");
        }
    }
    #[test]
    fn test_orthogonal_matrices_mean_zero() {
        let N = 10000;
        let M = 5;
        let device = tch::Device::Cuda(0);
        let random_matrices = random_orthogonal_matrices(N, M, device);
        
        let matrix_means = random_matrices.mean_dim(Some(&[0i64] as &[i64]), false, Kind::Float);
        let zeros = matrix_means.zeros_like();

        let total_diff_squared = tensor_diff_squared(&zeros, &matrix_means);
        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.001) {
            panic!("Generated orthogonal matrices do not have mean 0");
        }
    }
    #[test]
    fn test_orthogonal_matrices_singular_values() {
        let N = 10;
        let M = 10;
        let device = tch::Device::Cuda(0);
        let random_matrices = random_orthogonal_matrices(N, M, device);
        let (_, eigenvalues, _) = Tensor::linalg_svd(&random_matrices, true, "gesvd");

        let ones = eigenvalues.ones_like();

        let total_diff_squared = tensor_diff_squared(&ones, &eigenvalues);
        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.0001) {
            panic!("Generated orthogonal matrices do not have singular values of 1");
        }
    }
    #[test]
    fn test_log_normal_matrices_mean_zero() {
        let N = 10000;
        let M = 5;
        let log_normal_std_dev = 0.1;
        let device = tch::Device::Cuda(0);
        let random_matrices = random_log_normal_singular_value_matrices(N, M, log_normal_std_dev, device);
        
        let matrix_means = random_matrices.mean_dim(Some(&[0i64] as &[i64]), false, Kind::Float);
        let zeros = matrix_means.zeros_like();

        let total_diff_squared = tensor_diff_squared(&zeros, &matrix_means);
        println!("total squared diff: {}", total_diff_squared);
        if (total_diff_squared > 0.001) {
            panic!("Generated log-normal matrices do not have mean 0");
        }
    }
}
