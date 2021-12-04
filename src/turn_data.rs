use ndarray::*;
use rand::Rng;
use rand::seq::SliceRandom;

pub struct TurnData {
    pub flattened_matrix_set : Vec<Array1<f32>>,
    pub child_visit_probabilities : Array2<f32>,
    pub flattened_matrix_target : Array1<f32>
}
impl TurnData {
    pub fn permute<R : Rng + ?Sized>(mut self, rng : &mut R) -> Self {
        let k = self.flattened_matrix_set.len();
        let mut result_flattened_matrix_set = Vec::new();
        for _ in 0..k {
            result_flattened_matrix_set.push(Option::None);
        }
        let mut result_visit_probabilities = self.child_visit_probabilities.clone();

        let mut permutation : Vec<usize> = (0..k).collect();
        permutation.shuffle(rng);

        for (flattened_matrix, i) in self.flattened_matrix_set.drain(..).zip(0..k) {
            let dest_index = permutation[i];
            result_flattened_matrix_set[dest_index] = Option::Some(flattened_matrix);
        }

        let result_flattened_matrix_set = result_flattened_matrix_set.drain(..).map(|x| x.unwrap()).collect();

        for i in 0..k {
            let dest_i = permutation[i];
            for j in 0..k {
                let dest_j = permutation[j];
                result_visit_probabilities[[dest_i, dest_j]] = self.child_visit_probabilities[[i, j]];
            }
        }

        TurnData {
            flattened_matrix_set : result_flattened_matrix_set,
            child_visit_probabilities : result_visit_probabilities,
            flattened_matrix_target : self.flattened_matrix_target
        }
    }
}
