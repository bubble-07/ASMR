extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

use rand::Rng;
use rand::seq::SliceRandom;

use tch::{kind, Tensor};

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::vector_set::*;

#[derive(Clone)]
pub struct MatrixSet {
    pub matrices : Vec<Array2<f32>>
}

impl MatrixSet {
    pub fn get_flattened_tensors(&self) -> Vec<Tensor> {
        let mut result = Vec::new();
        for i in 0..self.size() {
            let flattened_matrix = flatten_matrix(self.matrices[i].view());
            let tensor = vector_to_tensor(flattened_matrix);
            result.push(tensor);
        }
        result
    }

    pub fn size(&self) -> usize {
        self.matrices.len()
    }

    pub fn get(&self, index : usize) -> ArrayView2<f32> {
        self.matrices[index].view()
    }

    pub fn get_newest_matrix(&self) -> ArrayView2<f32> {
        self.matrices[self.size() - 1].view()
    }

    pub fn add_matrix(self, matrix : Array2<f32>) -> MatrixSet {
        let mut matrices = self.matrices;
        matrices.push(matrix);
        MatrixSet {
            matrices
        }
    }

    pub fn shuffle<R : Rng + ?Sized>(&mut self, rng : &mut R) {
        self.matrices.shuffle(rng);
    }
    pub fn flatten(&self) -> VectorSet {
        let vectors = self.matrices.iter().map(|x| flatten_matrix(x.view()).to_owned()).collect();
        VectorSet {
            vectors
        }
    }
}
