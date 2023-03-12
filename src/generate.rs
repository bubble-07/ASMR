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
