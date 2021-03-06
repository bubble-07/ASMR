use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use crate::network::*;
use crate::neural_utils::*;
use crate::game_state::*;
use crate::array_utils::*;
use crate::params::*;
use crate::training_examples::*;
use ndarray::*;

use rand::Rng;
use rand::seq::SliceRandom;
use core::iter::Sum;

///K: Number of matrices in the initial set
///N: number of training examples
///M : matrix (flattened) dims
///F : feature dims
pub struct NetworkConfig {
    ///Injection network, taking matrix, target pairs, and turning them into initial features -- M x M -> F
    pub injector_net : BiConcatThenSequential,
    ///Main network - a stack of attentional residual layers interspersed with RELU. K x F -> (K + 1) x F
    pub main_net : ResidualAttentionStackWithGlobalTrack,
    ///Policy extraction network, taking "left", "right", 
    ///and "global" feature maps -- F x F x F -> F
    pub policy_extraction_net : TriConcatThenSequential
}

impl NetworkConfig {
    pub fn new(params : &Params, vs : &nn::Path) -> NetworkConfig {
        let injector_net = injector_net(params, vs / "injector");
        let main_net = main_net(params, vs / "main");
        let policy_extraction_net = policy_extraction_net(params, vs / "policy_extractor");
        NetworkConfig {
            injector_net,
            main_net,
            policy_extraction_net
        }
    }

    pub fn get_single_embeddings(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor)
                            -> Vec<Tensor> {
        let k = flattened_matrix_set.len();
        let mut single_embeddings = Vec::new();
        for i in 0..k {
            let flattened_matrix = &flattened_matrix_set[i];
            let embedding = self.injector_net.forward(flattened_matrix, flattened_matrix_target);
            single_embeddings.push(embedding);
        }
        single_embeddings
    }

    fn get_unnormalized_policy(&self, single_embeddings : &[Tensor]) -> Tensor {
        let k = single_embeddings.len();

        let mut averaged_embedding = Iterator::sum(single_embeddings.iter());
        averaged_embedding *= (1.0f32 / (k as f32));

        let mut main_net_inputs = Vec::new();
        for single_embedding in single_embeddings {
            main_net_inputs.push(single_embedding.shallow_clone());
        }
        main_net_inputs.push(averaged_embedding);

        let main_net_inputs = main_net_inputs;

        let mut main_net_outputs = self.main_net.forward(&main_net_inputs); 
        let main_net_global_output = main_net_outputs.pop().unwrap();
        //K x NxF
        let main_net_individual_outputs = main_net_outputs;
        
        let mut policy_tensors_by_left = Vec::new();
        //Compute policy logits for all left and right vectors
        for i in 0..k {
            let left_input = &main_net_individual_outputs[i];

            let mut policy_logits_for_left = Vec::new();
            for j in 0..k {
                let right_input = &main_net_individual_outputs[j]; 
                //Nx1
                let policy_logit = self.policy_extraction_net.forward(left_input, right_input, &main_net_global_output);
                policy_logits_for_left.push(policy_logit);
            }

            //NxK
            let policy_tensor_for_left = Tensor::concat(&policy_logits_for_left, 1);
            policy_tensors_by_left.push(policy_tensor_for_left);
        }
        //NxKxK
        let unnormalized_policy_mat = Tensor::stack(&policy_tensors_by_left, 1);
        unnormalized_policy_mat
    }

    ///Given a collection of K feature vector stacks from the injector_net, 
    ///(dimension NxF, N is the number of samples and F is the number of features), yields a computed Policy [Matrix stack
    ///of dimension NxKxK] tensor.
    pub fn get_policy(&self, single_embeddings : &[Tensor]) -> Tensor {
        let k = single_embeddings.len();
        //NxKxK
        let unnormalized_policy_mat = self.get_unnormalized_policy(single_embeddings);

        let n = unnormalized_policy_mat.size()[0];

        let flattened_unnormalized_policy_mat = unnormalized_policy_mat.reshape(&[n, (k * k) as i64]);
        let flattened_policy_mat =
                    flattened_unnormalized_policy_mat.softmax(1, Kind::Float);

        let policy_mat = flattened_policy_mat.reshape(&[n, k as i64, k as i64]);

        policy_mat
    }

    ///Given a collection of K [flattened] matrix stacks (dimension NxM, N is the
    ///number of samples and M is the flattened matrix dimension), and a flattened matrix target
    ///(dimension NxM), yields a computed Policy [Matrix stack of dimension NxKxK] tensor
    pub fn get_policy_from_scratch(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor,
                                   as_unnormalized : bool) -> Tensor {

        //First, compute the embeddings for every (input, target) pair
        let single_embeddings = self.get_single_embeddings(flattened_matrix_set, flattened_matrix_target);

        //Then compute the policy from those embeddings
        let policy_mat = if (as_unnormalized) {
            self.get_unnormalized_policy(&single_embeddings)
        } else {
            self.get_policy(&single_embeddings)
        };

        policy_mat
    }
    pub fn get_loss_from_scratch(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor,
                                 target_policy : &Tensor) -> Tensor {
        let computed_unnormalized_policy = self.get_policy_from_scratch(flattened_matrix_set, flattened_matrix_target, true);
        let n = computed_unnormalized_policy.size()[0];
        let k = computed_unnormalized_policy.size()[1];

        let flattened_unnormalized_policy = computed_unnormalized_policy.reshape(&[n as i64, (k * k) as i64]);
        let log_softmaxed = flattened_unnormalized_policy.log_softmax(1, Kind::Float);

        let one_over_n = 1.0f32 / (n as f32);
        let flattened_log_softmaxed = one_over_n * log_softmaxed.reshape(&[(n * k * k) as i64]);
        let flattened_target_policy = target_policy.reshape(&[(n * k * k) as i64]);
        let inner_product = flattened_target_policy.dot(&flattened_log_softmaxed);

        -inner_product
    }

    pub fn run_training_epoch<R : Rng + ?Sized>(&self, params : &Params, 
                              training_examples : &TrainingExamples, 
                              opt : &mut Optimizer,
                              device : Device,
                              rng : &mut R) -> (f64, f64) { //return value is training loss, validation loss
        let mut total_train_loss = 0f64;

        let set_sizings = training_examples.get_set_sizings();
        let validation_loss_weightings = training_examples.get_validation_loss_weightings(params, &set_sizings);
        let training_loss_weightings = training_examples.get_training_loss_weightings(params, &set_sizings);

        let num_iters = params.train_batches_per_save;

        //First train
        for _ in 0..num_iters {
            let mut iter_comparison = 0f64;
            let mut iter_loss = Tensor::scalar_tensor(0f64, (Kind::Float, device));
            for (set_sizing, loss_weighting) in set_sizings.iter().zip(training_loss_weightings.iter()) {
                let total_training_batches = training_examples.get_total_number_of_training_batches(params, *set_sizing);
                if (total_training_batches == 0) {
                    continue; //Sux to sux
                }
                let batch_index = rng.gen_range(0..total_training_batches);
                let batch_index = BatchIndex {
                    set_sizing : *set_sizing,
                    batch_index
                };

                let mut normalized_loss = batch_index.get_normalized_loss(params, training_examples, &self, device);
                normalized_loss *= *loss_weighting;
                iter_loss += normalized_loss;

                let mut normalized_loss_comparison = training_examples.get_normalized_random_choice_loss(*set_sizing);
                normalized_loss_comparison *= *loss_weighting as f64;
                iter_comparison += normalized_loss_comparison;
            }

            let float_iter_loss = f64::from(&iter_loss);
            println!("Iter loss: {}, Random choice loss: {}", float_iter_loss, iter_comparison);

            opt.backward_step(&iter_loss);
            total_train_loss += float_iter_loss / (num_iters as f64);
        }
        //Now compute validation loss
        let _guard = no_grad_guard();
        let mut validation_loss = 0f64;
        for relative_unwrapped_batch_index in 0..params.held_out_validation_batches {
            for (set_sizing, loss_weighting) in set_sizings.iter().zip(validation_loss_weightings.iter()) {
                let min_batch_index = training_examples.get_total_number_of_training_batches(params, *set_sizing);
                let num_validation_batches = training_examples.get_total_number_of_validation_batches(params, *set_sizing);
                let batch_index = min_batch_index + (relative_unwrapped_batch_index % num_validation_batches);
                let batch_index = BatchIndex {
                    set_sizing : *set_sizing,
                    batch_index
                };
                
                let mut normalized_loss = batch_index.get_normalized_loss(params, training_examples, &self, device);
                normalized_loss *= *loss_weighting;
                
                validation_loss += f64::from(&normalized_loss);
            }
        }
        validation_loss /= params.held_out_validation_batches as f64;

        (total_train_loss, validation_loss)
    }
}
