
pub struct TrainingExample {
    flattened_matrix_set : Vec<Array1<f32>>,
    flattened_matrix_target : Array1<f32>,
    child_visit_probabilities : Array2<f32>
}
