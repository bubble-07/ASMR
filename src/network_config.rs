use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind};
use crate::network::*;
use crate::neural_utils::*;
use crate::game_state::*;
use crate::array_utils::*;
use crate::params::*;
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
    right_policy_extraction_net : ConcatThenSequential
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
        NetworkConfig {
            injector_net,
            combiner_net,
            left_policy_extraction_net,
            right_policy_extraction_net
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

        let policy_tensor = self.get_policy(&single_embeddings, &combined_embedding, false);
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

    ///Given a collection of K feature vector stacks (dimension NxF, N is the number
    ///of samples and F is the number of features), and a feature vector stack for their
    ///combined embedding (dimension NxF), yields a computed Policy [Matrix stack
    ///of dimension NxKxK] tensor. The final argument determines if the policy
    ///is expressed in terms of probabilities or in terms of log-probabilities
    fn get_policy(&self, single_embeddings : &[Tensor], combined_embedding : &Tensor,
                  as_log : bool) -> Tensor {
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
        let unnormalized_policy_mat = left_policy_mat.matmul(&right_policy_mat);

        let n = unnormalized_policy_mat.size()[0];

        let flattened_unnormalized_policy_mat = unnormalized_policy_mat.reshape(&[n, (k * k) as i64]);
        let flattened_policy_mat = if (as_log) {
            flattened_unnormalized_policy_mat.log_softmax(1, Kind::Float)
        } else {
            flattened_unnormalized_policy_mat.softmax(1, Kind::Float)
        };
        let policy_mat = flattened_policy_mat.reshape(&[n, k as i64, k as i64]);

        policy_mat
    }

    ///Given a collection of K [flattened] matrix stacks (dimension NxM, N is the
    ///number of samples and M is the flattened matrix dimension), and a flattened matrix target
    ///(dimension NxM), yields a computed Policy [Matrix stack of dimension NxKxK] tensor
    pub fn get_policy_from_scratch(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor,
                                   as_log : bool) -> Tensor {

        //First, compute the embeddings for every input matrix
        let single_embeddings = self.get_single_embeddings(flattened_matrix_set, flattened_matrix_target);

        //Then, combine embeddings repeatedly until there's only one "master" embedding for the
        //entire collection of matrices
        let combined_embedding = self.combine_embeddings(&single_embeddings);

        let policy_mat = self.get_policy(&single_embeddings, &combined_embedding, as_log);

        policy_mat
    }
    pub fn get_loss_from_scratch(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor,
                                 target_policy : &Tensor) -> Tensor {
        let computed_log_policy = self.get_policy_from_scratch(flattened_matrix_set, flattened_matrix_target, true);
        let n = computed_log_policy.size()[0];
        let k = computed_log_policy.size()[1];
        let d = n * k * k;

        let flattened_log_policy = computed_log_policy.reshape(&[d]);
        let flattened_target_policy = target_policy.reshape(&[d]);
        let negative_loss = flattened_target_policy.dot(&flattened_log_policy);
        let loss = -negative_loss;
        loss
    }
}
