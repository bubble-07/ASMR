extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Uniform, Distribution, WeightedIndex};
use std::convert::{TryFrom, TryInto};

use tch::{kind, Tensor};

pub fn tensor_to_vector(tensor : &Tensor) -> Array1<f32> {
    ArrayBase::from_vec(tensor.into())
}

pub fn tensor_to_matrix(tensor : &Tensor) -> Array2<f32> {
    let shape : Vec<usize> = tensor.size().iter().map(|s| *s as usize).collect();
    tensor_to_vector(tensor).into_shape(shape).unwrap().into_dimensionality::<Ix2>().unwrap()
}

pub fn vector_to_tensor(vec : ArrayView1<f32>) -> Tensor {
    Tensor::try_from(vec.to_slice().unwrap()).unwrap()
}

pub fn matrix_to_tensor(mat : ArrayView2<f32>) -> Tensor {
    let shape : Vec<i64> = mat.shape().iter().map(|s| *s as i64).collect();
    let flat_mat = flatten_matrix(mat);
    let tensor = vector_to_tensor(flat_mat);
    tensor.reshape(&shape)
}

pub fn sq_frob_dist(a : ArrayView2<f32>, b : ArrayView2<f32>) -> f32 {
    let diff = &a - &b;
    let flattened_diff = flatten_matrix(diff.view());
    let ret = flattened_diff.dot(&flattened_diff);
    ret
}

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

pub fn sample_index_pair<R : Rng + ?Sized>(mat : ArrayView2<f32>, rng : &mut R) -> (usize, usize) {
    let cols = mat.shape()[1];
    let mat_as_slice = mat.to_slice().unwrap();
    let dist = WeightedIndex::new(mat_as_slice).unwrap();
    let flattened_index = dist.sample(rng);

    let col_index = flattened_index % cols;
    let row_index = (flattened_index - col_index) / cols;
    (row_index, col_index)
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
