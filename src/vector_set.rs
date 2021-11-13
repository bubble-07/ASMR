extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

use rand::Rng;
use rand::seq::SliceRandom;

use ndarray::*;
use ndarray_linalg::*;

#[derive(Clone)]
pub struct VectorSet {
    pub vectors : Vec<Array1<f32>>
}

impl VectorSet {
    pub fn shuffle<R : Rng + ?Sized>(&mut self, rng : &mut R) {
        self.vectors.shuffle(rng);
    }
}
