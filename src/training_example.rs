use ndarray::*;

pub struct TrainingExample {
    flattened_matrix_set : Vec<Array1<f32>>,
    flattened_matrix_target : Array1<f32>,
    child_visit_probabilities : Array2<f32>
}

///Grouped training examples for a single game-tree
pub struct TrainingExamples {
    pub flattened_matrix_sets : Vec<Vec<Array1<f32>>>,
    pub flattened_matrix_target : Array1<f32>,
    pub child_visit_probabilities : Vec<Array2<f32>>
}
