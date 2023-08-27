extern crate ndarray;

use ndarray::*;

use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Uniform, Distribution, WeightedIndex};
use std::convert::{TryFrom, TryInto};

use tch::{kind::Kind, Tensor, Device};

pub fn tensor_diff_squared(one : &Tensor, two : &Tensor) -> f32 {
    let diff = one - two;
    let diff_squared = &diff * &diff;
    let total_diff_squared = f32::from(diff_squared.sum(Kind::Float));
    total_diff_squared
}

pub fn generate_2d_index_tensor_span(num_matrices : i64, device : Device) -> (Tensor, Tensor) {
    let num_children = num_matrices * num_matrices;

    let left_indices = Tensor::arange(num_matrices, (Kind::Int64, device));
    let left_indices = left_indices.repeat(&[num_matrices]);

    let right_indices = Tensor::arange(num_matrices, (Kind::Int64, device));
    let right_indices = right_indices.repeat_interleave_self_int(num_matrices, Option::None,
                                                                 Option::Some(num_children));
    (left_indices, right_indices)
}
