extern crate ndarray;
extern crate ndarray_linalg;
extern crate rand;

use rand::Rng;
use rand::seq::SliceRandom;

use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};

use serde::{Serialize, Deserialize};
use ndarray::*;
use ndarray_linalg::*;
use crate::array_utils::*;
use std::fmt;

///R x (k + t) x (k + t)
pub struct VisitLogitMatrices(pub Tensor);

impl VisitLogitMatrices {
    pub fn split(self, split_sizes : &[i64]) -> Vec<VisitLogitMatrices> {
        let mut visit_logit_matrices = self.0.split_with_sizes(split_sizes, 0);
        visit_logit_matrices.drain(..).map(|x| VisitLogitMatrices(x)).collect()
    }
    pub fn merge(mut visit_logit_matrices : Vec<VisitLogitMatrices>) -> VisitLogitMatrices {
        let visit_logit_matrices : Vec<Tensor> = visit_logit_matrices.drain(..).map(|x| x.0).collect();
        let visit_logit_matrices = Tensor::concat(&visit_logit_matrices, 0);
        VisitLogitMatrices(visit_logit_matrices)
    }

    pub fn get_num_matrices(&self) -> i64 {
        let VisitLogitMatrices(child_visit_logits) = &self;
        child_visit_logits.size()[0]
    }
    pub fn get_matrix_dim(&self) -> i64 {
        let VisitLogitMatrices(child_visit_logits) = &self;
        child_visit_logits.size()[1]
    }

    pub fn get_normalized_random_choice_loss(&self) -> f64 {
        let k_plus_t = self.get_matrix_dim();
        let k_plus_t_squared = k_plus_t * k_plus_t;

        let nat_log = (k_plus_t_squared as f64).ln();
        nat_log
    }

    //Returns a normalized collection of probability matrices for policies
    pub fn get_policy(&self) -> Tensor { 
        let r = self.get_num_matrices();
        let k_plus_t = self.get_matrix_dim();

        let flattened_policy = self.get_flattened_policy();
        flattened_policy.reshape(&[r, k_plus_t, k_plus_t])
    }
    //Gets the cross-entropy-with-logits loss for _just_ across the largest-indexed
    //row and column, treating the rest of the matrix as one block whose
    //contributions are pooled, but re-normalized to remove any bias from
    //this extra fake element.
    pub fn get_peel_loss(&self, target_policy : &Tensor, label_smoothing_factor : f64) -> Tensor {
        let VisitLogitMatrices(child_visit_logits) = &self;

        let s = child_visit_logits.size();
        let (r, k_plus_t) = (s[0], s[1]);
        let ind = k_plus_t - 1;
        //R x K+t for all of these
        let last_row_logits = child_visit_logits.i((.., ind, ..));
        let last_col_logits = child_visit_logits.i((.., .., ind));

        let last_row_policy = target_policy.i((.., ind, ..));
        let last_col_policy = target_policy.i((.., .., ind));
        
        //R x 2*(K+t), last element is technically a duplicate
        let all_logits = Tensor::concat(&[last_row_logits, last_col_logits], 1);
        let all_policy = Tensor::concat(&[last_row_policy, last_col_policy], 1);

        //Replace the very last element with probability-sums over the interior
        //matrices not including the last index. For the logits matrix,
        //this will mean using logsumexp instead.
        //R x K+t-1 x K+t-1
        let logits_submatrices = child_visit_logits.slice(1, Option::None, Option::Some(ind), 1);
        let logits_submatrices = logits_submatrices.slice(2, Option::None, Option::Some(ind), 1);

        let policy_submatrices = target_policy.slice(1, Option::None, Option::Some(ind), 1);
        let policy_submatrices = policy_submatrices.slice(2, Option::None, Option::Some(ind), 1);

        //R
        let logits_submatrix_sums = logits_submatrices.logsumexp(&[1, 2], false);
        let policy_submatrix_sums = policy_submatrices.sum_to_size(&[r, 1, 1]).reshape(&[r]);

        //Now, replace the appropriate elements
        let batch_indices = Tensor::arange(r, (Kind::Int64, target_policy.device()));
        let batch_indices = ((2 * k_plus_t) as i64) * batch_indices + ind;

        let all_logits = all_logits.put(&batch_indices, &logits_submatrix_sums, false);
        let all_policy = all_policy.put(&batch_indices, &policy_submatrix_sums, false);
        
        //With that info, output a loss
        let unnormalized_loss = Self::softmax_cross_entropy_with_logits(&all_logits, &all_policy,
                                                                        label_smoothing_factor);
        let normalization_factor = (((2 * k_plus_t - 1) as f64) / ((2 * k_plus_t) as f64));
        let loss = normalization_factor * unnormalized_loss;
        loss
    }

    pub fn get_loss(&self, target_policy : &Tensor, label_smoothing_factor : f64) -> Tensor {
        let VisitLogitMatrices(child_visit_logits) = &self;
        Self::softmax_cross_entropy_with_logits(child_visit_logits, target_policy,
                                                label_smoothing_factor)
    }

    //Assuming the first dimension is the batch dimension, compute cross-entropy with logits
    //normalized by the number of samples in the batch.
    fn softmax_cross_entropy_with_logits(logits : &Tensor, target_policy : &Tensor,
                                         label_smoothing_factor : f64) -> Tensor {
        let r = logits.size()[0];
        let total_size = logits.size().drain(..).fold(1, |a, b| a * b);
        let remainder_size = total_size / r;

        let one_over_r = 1.0f64 / (r as f64);

        let nonflat_dim = [r, remainder_size];
        let flat_dim = [total_size];

        let reshaped_logits = logits.reshape(&nonflat_dim);
        let log_softmaxed = reshaped_logits.log_softmax(1, Kind::Float);

        let flattened_log_softmaxed = one_over_r * log_softmaxed.reshape(&flat_dim);
        let flattened_target_policy = target_policy.reshape(&flat_dim);

        //Smooth the target policy
        let uniform_element_value = 1.0f64 / (remainder_size as f64);
        let smoothed_flattened_policy = (1.0f64 - label_smoothing_factor) * flattened_target_policy +
                                        (label_smoothing_factor * uniform_element_value);


        let inner_product = smoothed_flattened_policy.dot(&flattened_log_softmaxed);

        -inner_product
    }

    fn get_flattened_policy(&self) -> Tensor {
        let r = self.get_num_matrices();
        let k_plus_t = self.get_matrix_dim();
        let VisitLogitMatrices(child_visit_logits) = &self;

        let flattened_child_visit_logits = child_visit_logits.reshape(&[r, k_plus_t * k_plus_t]);
        let flattened_policy = flattened_child_visit_logits.softmax(1, Kind::Float); 
        flattened_policy
    }
    
    //Draw index samples from the policy matrix -- 1D shape R, 
    //indices are in (k + t) * (k + t)
    pub fn draw_indices(&self) -> (Tensor, Tensor) {
        let _guard = no_grad_guard();
        let r = self.get_num_matrices();
        let k_plus_t = self.get_matrix_dim();

        let flattened_policy = self.get_flattened_policy();

        //Draw index samples from the policy matrix -- 1D shape R, 
        //indices are in (k + t) * (k + t)
        let sampled_joint_indices = flattened_policy.multinomial(1, true).reshape(&[r]);
        let left_indices = sampled_joint_indices.divide_scalar_mode(k_plus_t, "trunc");
        let right_indices = sampled_joint_indices.fmod(k_plus_t);

        (left_indices, right_indices)
    }
    //Masks out the chosen moves from the matrices with a -inf
    pub fn mask_chosen(&mut self, left_indices : &Tensor, right_indices : &Tensor) {
        let _guard = no_grad_guard();
        let r = self.get_num_matrices();
        let k_plus_t = self.get_matrix_dim();

        let row_step = k_plus_t;
        let batch_step = k_plus_t * k_plus_t;


        let VisitLogitMatrices(matrices) = self;


        let left_offsets = left_indices * row_step;
        let matrix_offsets = left_offsets + right_indices;

        let batch_indices = Tensor::arange(r, (Kind::Int64, matrices.device()));
        let batch_offsets = batch_step * batch_indices;

        let offsets = batch_offsets + matrix_offsets;


        let mut values = Tensor::zeros(&[r], (Kind::Float, matrices.device()));
        let values = values.fill_(f64::NEG_INFINITY);

        let result = matrices.put(&offsets, &values, false);

        self.0 = result;
    }
}
