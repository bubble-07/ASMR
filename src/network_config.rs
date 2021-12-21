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

///M : matrix (flattened) dims
///F : feature dims
pub struct NetworkConfig {
    ///M x M -> F (single)
    injector_net : ConcatThenSequential,
    ///F x F -> F (combined)
    combiner_net : ConcatThenSequential,
    ///F (combined) x F (single) -> 2F
    left_policy_extraction_net : ConcatThenSequential,
    ///F (combined) x F (single) -> 2F
    right_policy_extraction_net : ConcatThenSequential,
    ///Single scalar to scale the unnormalized policy after the dot-products
    policy_confidence_scaling : Tensor
}

pub struct RolloutState {
    pub game_state : GameState,
    single_embeddings : Vec<Tensor>,
    combined_embedding : Tensor
}

impl Clone for RolloutState {
    fn clone(&self) -> Self {
        let single_embeddings = self.single_embeddings.iter().map(|x| x.copy()).collect();
        RolloutState {
            game_state : self.game_state.clone(),
            single_embeddings,
            combined_embedding : self.combined_embedding.copy()
        }
    }
}

impl NetworkConfig {
    pub fn new(params : &Params, vs : &nn::Path) -> NetworkConfig {
        let injector_net = injector_net(params, vs / "injector");
        let combiner_net = combiner_net(params, vs / "combiner");
        let left_policy_extraction_net = half_policy_extraction_net(params, vs / "left_policy_vector_supplier");
        let right_policy_extraction_net = half_policy_extraction_net(params, vs / "right_policy_vector_supplier");
        let policy_confidence_scaling = vs.var("policy_confidence_scaling", &[], 
                                               Init::Uniform {lo : 0.5, up : 1.0});
        NetworkConfig {
            injector_net,
            combiner_net,
            left_policy_extraction_net,
            right_policy_extraction_net,
            policy_confidence_scaling
        }
    }
    pub fn start_rollout(&self, game_state : GameState) -> RolloutState {
        let _guard = no_grad_guard();

        let flattened_matrix_set = game_state.get_flattened_matrix_set();
        let flattened_matrix_target = game_state.get_flattened_matrix_target();

        let single_embeddings = self.get_single_embeddings(&flattened_matrix_set, &flattened_matrix_target);
        let combined_embedding = self.combine_embeddings(&single_embeddings);

        RolloutState {
            game_state,
            single_embeddings,
            combined_embedding
        }
    }

    pub fn manual_step_rollout(&self, rollout_state : RolloutState, new_mat : Array2<f32>) -> RolloutState {
        let _guard = no_grad_guard();

        let mut single_embeddings = rollout_state.single_embeddings;
        let mut combined_embedding = rollout_state.combined_embedding;
         
        let game_state = rollout_state.game_state.add_matrix(new_mat);

        let flattened_matrix_target = game_state.get_flattened_matrix_target();
       
        let new_mat = game_state.get_newest_matrix();
        let new_tensor = vector_to_tensor(flatten_matrix(new_mat.view()));
        let new_embedding = self.injector_net.forward(&new_tensor, &flattened_matrix_target);

        combined_embedding = self.combiner_net.forward(&combined_embedding, &new_embedding);

        single_embeddings.push(new_embedding);

        RolloutState {
            game_state,
            single_embeddings,
            combined_embedding
        }
    }

    fn step_rollout<R : Rng + ?Sized>(&self, rollout_state : RolloutState, rng : &mut R) -> RolloutState {
        let _guard = no_grad_guard();

        let single_embeddings = &rollout_state.single_embeddings;
        let combined_embedding = &rollout_state.combined_embedding;

        let policy_tensor = self.get_policy(&single_embeddings, &combined_embedding);
        let policy_mat = tensor_to_matrix(&policy_tensor);
        let (ind_one, ind_two) = sample_index_pair(policy_mat.view(), rng);

        let game_state = &rollout_state.game_state;

        let mat_one = game_state.get_matrix(ind_one);
        let mat_two = game_state.get_matrix(ind_two);

        let new_mat = mat_one.dot(&mat_two);
        
        self.manual_step_rollout(rollout_state, new_mat)
    }
    
    pub fn complete_rollout<R : Rng + ?Sized>(&self, mut rollout_state : RolloutState, rng : &mut R) -> f32 {
        let turns = rollout_state.game_state.get_remaining_turns();
        for _ in 0..turns {
            rollout_state = self.step_rollout(rollout_state, rng);
        }
        let value = rollout_state.game_state.get_distance();
        value
    }

    pub fn perform_rollout<R : Rng + ?Sized>(&self, game_state : GameState, rng : &mut R) -> f32 {
        let rollout_state = self.start_rollout(game_state);
        self.complete_rollout(rollout_state, rng)
    }

    ///Given embeddings of dimension F, yields a combined embedding of dimension F
    fn combine_embeddings(&self, embeddings : &[Tensor]) -> Tensor {
        let k = embeddings.len();
        if k == 1 {
            embeddings[0].shallow_clone()
        } else {
            let mut updated = Vec::new();
            for i in 0..k {
                let elem = embeddings[i].shallow_clone();
                if i % 2 == 0 {
                    updated.push(elem);
                } else {
                    let prev_elem = updated.pop().unwrap();
                    let combined = self.combiner_net.forward(&prev_elem, &elem);
                    updated.push(combined);
                }
            }
            self.combine_embeddings(&updated)
        }
    }

    fn get_single_embeddings(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor)
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

    fn get_unnormalized_policy(&self, single_embeddings : &[Tensor], combined_embedding : &Tensor) -> Tensor {
        let k = single_embeddings.len();
        //Compute left and right policy vectors
        let mut left_policy_vecs = Vec::new();
        let mut right_policy_vecs = Vec::new();
        for i in 0..k {
            let single_embedding = &single_embeddings[i];
            let left_policy_vec = self.left_policy_extraction_net.forward(&combined_embedding, single_embedding);
            let right_policy_vec = self.right_policy_extraction_net.forward(&combined_embedding, single_embedding);
            left_policy_vecs.push(left_policy_vec);
            right_policy_vecs.push(right_policy_vec);
        }

        //Combine the left and right policy vectors by doing pairwise dot-products
        //Each policy vec is NxF
        //NxKxF
        let left_policy_mat = Tensor::stack(&left_policy_vecs, 1);
        //NxFxK
        let right_policy_mat = Tensor::stack(&right_policy_vecs, 2);
        //NxKxK
        
        let mut unnormalized_policy_mat = left_policy_mat.matmul(&right_policy_mat);
        unnormalized_policy_mat *= &(1.0 + self.policy_confidence_scaling.celu());
        unnormalized_policy_mat
    }

    ///Given a collection of K feature vector stacks (dimension NxF, N is the number
    ///of samples and F is the number of features), and a feature vector stack for their
    ///combined embedding (dimension NxF), yields a computed Policy [Matrix stack
    ///of dimension NxKxK] tensor. The final argument determines if the policy
    ///is expressed in terms of probabilities or in terms of log-probabilities
    fn get_policy(&self, single_embeddings : &[Tensor], combined_embedding : &Tensor) -> Tensor {
        let k = single_embeddings.len();
        //NxKxK
        let unnormalized_policy_mat = self.get_unnormalized_policy(single_embeddings, combined_embedding);

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

        //First, compute the embeddings for every input matrix
        let single_embeddings = self.get_single_embeddings(flattened_matrix_set, flattened_matrix_target);

        //Then, combine embeddings repeatedly until there's only one "master" embedding for the
        //entire collection of matrices
        let combined_embedding = self.combine_embeddings(&single_embeddings);

        let policy_mat = if (as_unnormalized) {
            self.get_unnormalized_policy(&single_embeddings, &combined_embedding)
        } else {
            self.get_policy(&single_embeddings, &combined_embedding)
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
