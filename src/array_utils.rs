extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

///Returns `true` only if all elements of `vec` are finite floats.
pub fn all_finite(vec : ArrayView1<f32>) -> bool {
    let n = vec.shape()[0];
    for i in 0..n {
        if !vec[[i,]].is_finite() {
            return false;
        }
    }
    true
}

///Vectorizes (flattens) the given matrix `mat`.
pub fn flatten_matrix(mat : ArrayView2<f32>) -> ArrayView1<f32> {
    let full_dim = mat.shape()[0] * mat.shape()[1];
    let reshaped = mat.clone().into_shape((full_dim,)).unwrap();
    reshaped
}

///Given a vector of floats, yields the index and the value of the largest
///float in `vec`.
pub fn max_index_and_value(vec : ArrayView1<f32>) -> (usize, f32) {
    let mut max_index = 0; 
    let mut max_value = vec[[0,]];
    for i in 1..vec.shape()[0] {
        if vec[[i,]] > max_value {
            max_value = vec[[i,]];
            max_index = i;
        }
    }
    (max_index, max_value)
}
