use ndarray::*;
use serde::{Serialize, Deserialize};

///Grouped game-data for a single game-tree
#[derive(Serialize, Deserialize)]
pub struct GameData {
    pub flattened_matrix_sets : Vec<Vec<Array1<f32>>>,
    pub flattened_matrix_target : Array1<f32>,
    pub child_visit_probabilities : Vec<Array2<f32>>
}
