use crate::matrix_set::*;
use crate::array_utils::*;
use std::cmp::min;
use tch::{kind, Tensor};
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

#[derive(Clone)]
pub struct GameState {
    pub matrix_set : MatrixSet,
    pub target : Array2<f32>,
    pub remaining_turns : usize,
    pub distance : f32
}

impl GameState {
    pub fn new(matrix_set : MatrixSet, target : Array2<f32>, remaining_turns : usize) -> Self {
        let mut distance = f32::MAX;
        for i in 0..matrix_set.size() {
            let matrix = matrix_set.get(i);
            let matrix_distance = sq_frob_dist(matrix.view(), target.view());
            distance = distance.min(matrix_distance);
        }
        GameState {
            matrix_set,
            target,
            remaining_turns,
            distance
        }
    }

    pub fn add_matrix(self, matrix : Array2<f32>) -> Self {
        let matrix_distance = sq_frob_dist(matrix.view(), self.target.view());

        let matrix_set = self.matrix_set.add_matrix(matrix);
        let distance = self.distance.min(matrix_distance);

        let remaining_turns = self.remaining_turns - 1;
        GameState {
            matrix_set,
            target : self.target,
            remaining_turns,
            distance
        }
    }

    pub fn get_flattened_matrix_target(&self) -> Tensor {
        let flattened = flatten_matrix(self.target.view());
        vector_to_tensor(flattened)
    }

    pub fn get_flattened_matrix_set(&self) -> Vec<Tensor> {
        self.matrix_set.get_flattened_tensors()
    }

    pub fn get_target(&self) -> ArrayView2<f32> {
        self.target.view()
    }
    pub fn get_matrix_set(&self) -> &MatrixSet {
        &self.matrix_set
    }
    pub fn get_newest_matrix(&self) -> ArrayView2<f32> {
        self.matrix_set.get_newest_matrix()
    }
    pub fn get_num_matrices(&self) -> usize {
        self.matrix_set.size()
    }
    pub fn get_remaining_turns(&self) -> usize {
        self.remaining_turns
    }
    pub fn get_distance(&self) -> f32 {
        self.distance
    }
    pub fn get_matrix(&self, ind : usize) -> ArrayView2<f32> {
        self.matrix_set.get(ind)
    }
    pub fn perform_move(self, ind_one : usize, ind_two : usize) -> Self {
        let matrix = self.matrix_set.get(ind_one).dot(&self.matrix_set.get(ind_two));
        self.add_matrix(matrix)
    }
}
