use ndarray::*;
use serde::{Serialize, Deserialize};
use crate::turn_data::*;

///Grouped game-data for a single game-tree
#[derive(Serialize, Deserialize)]
pub struct GameData {
    pub flattened_matrix_sets : Vec<Vec<Array1<f32>>>,
    pub flattened_matrix_target : Array1<f32>,
    pub child_visit_probabilities : Vec<Array2<f32>>
}

impl GameData {
    pub fn get_turn_data(mut self) -> Vec<TurnData> {
        let mut result = Vec::new();
        for (flattened_matrix_set, child_visit_probabilities) in 
            self.flattened_matrix_sets.drain(..).zip(self.child_visit_probabilities.drain(..)) {
            let turn_data = TurnData {
                flattened_matrix_set,
                child_visit_probabilities,
                flattened_matrix_target : self.flattened_matrix_target.clone()
            };
            result.push(turn_data);
        }
        result
    }
}
