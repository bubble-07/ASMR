extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

use rand::Rng;
use rand::seq::SliceRandom;

use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use crate::vector_set::*;

#[derive(Clone)]
pub struct MatrixSet {
    pub matrices : Vec<Array2<f32>>
}

impl MatrixSet {
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
