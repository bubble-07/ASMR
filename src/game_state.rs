use crate::matrix_set::*;
use crate::array_utils::*;
use std::cmp::min;
use rand::Rng;
use tch::{kind, Tensor};
use serde::{Serialize, Deserialize};
use std::fmt;
extern crate ndarray;
extern crate ndarray_linalg;

use ndarray::*;

#[derive(Clone, Serialize, Deserialize)]
pub struct GameState {
    pub matrix_set : MatrixSet,
    pub target : Array2<f32>,
    pub remaining_turns : usize,
    pub distance : f32
}

impl fmt::Display for GameState {
    fn fmt(&self, f : &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "set: {} \n target : {} \n turns : {}", &self.matrix_set, &self.target, self.remaining_turns)
    }
}

impl GameState {
    pub fn new(matrix_set : MatrixSet, target : Array2<f32>, remaining_turns : usize) -> Self {
        let mut distance = f32::MAX;
        let MatrixSet(matrices) = &matrix_set;
        for matrix in matrices {
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

    pub fn get_target(&self) -> ArrayView2<f32> {
        self.target.view()
    }
    pub fn get_matrix_set(&self) -> &MatrixSet {
        &self.matrix_set
    }
    pub fn get_remaining_turns(&self) -> usize {
        self.remaining_turns
    }
    pub fn get_distance(&self) -> f32 {
        self.distance
    }
    pub fn perform_move(self, ind_one : usize, ind_two : usize) -> Self {
        let matrix = self.matrix_set.get(ind_one).dot(&self.matrix_set.get(ind_two));
        self.add_matrix(matrix)
    }
    pub fn complete_random_rollout<R : Rng + ?Sized>(self, rng : &mut R) -> f32 {
        let turns = self.get_remaining_turns();
        if (turns == 0) {
            self.get_distance()
        } else {
            let size = self.matrix_set.len();
            let left_ind = rng.gen_range(0..size);
            let right_ind = rng.gen_range(0..size);
            self.perform_move(left_ind, right_ind).complete_random_rollout(rng)
        }
    }
    pub fn complete_greedy_rollout(self) -> f32 {
        let turns = self.get_remaining_turns();
        if (turns == 0) {
            self.get_distance()
        } else {
            let mut min_distance = f32::INFINITY;
            let mut min_index_pair = (0, 0);

            let size = self.matrix_set.len();
            for ind_one in 0..size {
                for ind_two in 0..size {
                    let matrix = self.matrix_set.get(ind_one).dot(&self.matrix_set.get(ind_two));
                    let matrix_distance = sq_frob_dist(matrix.view(), self.target.view());
                    if (matrix_distance < min_distance) {
                        min_distance = matrix_distance;
                        min_index_pair = (ind_one, ind_two);
                    }
                }
            }
            let (left_ind, right_ind) = min_index_pair;
            self.perform_move(left_ind, right_ind).complete_greedy_rollout()
        }
    }
}
