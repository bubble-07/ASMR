extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

use rand::Rng;
use rand::seq::SliceRandom;

use tch::{kind, Tensor};

use serde::{Serialize, Deserialize};
use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use std::fmt;

#[derive(Clone, Serialize, Deserialize)]
pub struct MatrixSet(pub Vec<Array2<f32>>);

impl fmt::Display for MatrixSet {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        let MatrixSet(matrices) = &self;
        write!(f, "{:?}", matrices)
    }
}
impl MatrixSet {
    pub fn apply_orthonormal_basis_change(self, Q : ArrayView2<f32>) -> Self {
        let MatrixSet(mut matrices) = self;
        let result = matrices.drain(..)
                             .map(|x| Q.t().dot(&x).dot(&Q))
                             .collect();
        MatrixSet(result)
    }
    pub fn to_flattened_vec(self) -> Vec<f32> {
        let MatrixSet(mut matrices) = self;
        let mut result = Vec::new();
        for matrix in matrices.drain(..) {
            let mut flattened_matrix = matrix.into_raw_vec();
            result.append(&mut flattened_matrix);
        }
        result
    }
    pub fn len(&self) -> usize {
        let MatrixSet(matrices) = &self;
        matrices.len()
    }
    pub fn get(&self, index : usize) -> ArrayView2<f32> {
        let MatrixSet(matrices) = &self;
        matrices[index].view()
    }

    pub fn get_newest_matrix(&self) -> ArrayView2<f32> {
        let index = self.len() - 1;
        self.get(index)
    }

    pub fn add_matrix(self, matrix : Array2<f32>) -> Self {
        let MatrixSet(mut matrices) = self;
        matrices.push(matrix);
        MatrixSet(matrices)
    }
}
