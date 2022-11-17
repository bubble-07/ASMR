use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device, nn::VarStore};
use crate::network::*;
use crate::neural_utils::*;
use crate::game_state::*;
use crate::array_utils::*;
use crate::params::*;
use crate::batch_split_training_examples::*;
use crate::training_examples::*;
use crate::network_module::*;
use crate::network_rollout::*;
use crate::rollout_states::*;
use crate::bunched_rollout::*;
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
    ///Root network - a stack of attentional residual layers interspersed with RELU. K x F -> (K + 1) x F
    pub root_net : ResidualAttentionStackWithGlobalTrack,
    ///Peel network
    pub peel_net : PeelStack,
    ///Policy extraction network, taking "left", "right", 
    ///and "global" feature maps -- F x F x F -> F
    pub policy_extraction_net : TriConcatThenSequential
}

impl NetworkConfig {
    pub fn new(params : &Params, vs : &nn::Path) -> NetworkConfig {
        let injector_net = injector_net(params, vs / "injector");
        let root_net = root_net(params, vs / "root");
        let peel_net = peel_net(params, vs / "peel");
        let policy_extraction_net = policy_extraction_net(params, vs / "policy_extractor");
        NetworkConfig {
            injector_net,
            root_net,
            peel_net,
            policy_extraction_net
        }
    }

    pub fn get_single_embedding(&self, flattened_transformed_matrices : &Tensor, 
                                transformed_flattened_targets : &Tensor)
                            -> Tensor {
        self.injector_net.forward(flattened_transformed_matrices, &transformed_flattened_targets)
    }

    pub fn get_single_embeddings(&self, flattened_transformed_matrix_sets : &[Tensor], 
                                 transformed_flattened_targets : &Tensor)
                            -> Vec<Tensor> {
        flattened_transformed_matrix_sets.iter()
            .map(|x| self.get_single_embedding(x, transformed_flattened_targets))
            .collect()
    }

    ///Returns: pre-activations for each layer, 
    ///followed by the global output embedding and output embeddings for each matrix
    ///in the initial matrix set
    pub fn get_main_net_outputs(&self, single_embeddings : &[Tensor]) -> (Vec<Tensor>, Tensor, Vec<Tensor>) {
        let k = single_embeddings.len();

        let mut averaged_embedding : Tensor = Iterator::sum(single_embeddings.iter());
        averaged_embedding *= (1.0f32 / (k as f32));

        let mut main_net_inputs = Vec::new();
        for single_embedding in single_embeddings {
            main_net_inputs.push(single_embedding.shallow_clone());
        }
        main_net_inputs.push(averaged_embedding);

        //Nx(K+1)xF
        let main_net_inputs = Tensor::stack(&main_net_inputs, 1);

        //L x Nx(K+1)xF, global output in the last place of each feature map
        let mut main_activations = self.root_net.forward(&main_net_inputs); 

        //Pull out last-layer activations
        //Nx(K+1)xF
        let main_net_outputs = main_activations.pop().unwrap();

        //Split into (K+1) NxF feature maps
        let mut main_net_outputs = main_net_outputs.unbind(1);
        //Pull out the global output from the collection of outputs
        let main_net_global_output = main_net_outputs.pop().unwrap();

        (main_activations, main_net_global_output, main_net_outputs)
    }

    pub fn get_policy_logits(&self, main_net_global_output : &Tensor,
                             main_net_individual_outputs : &[Tensor]) -> Tensor {
        let k = main_net_individual_outputs.len();
        
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
        let logit_policy_mat = Tensor::stack(&policy_tensors_by_left, 1);
        logit_policy_mat
    }

    pub fn run_training_epoch<R : Rng + ?Sized>(&self, params : &Params, 
                              training_examples : &BatchSplitTrainingExamples, 
                              opt : &mut Optimizer,
                              device : Device,
                              rng : &mut R
                              ) -> (f64, f64) { //return value is training loss, validation loss

        let mut total_train_loss = 0f64;
        let num_iters = params.train_batches_per_save;

        //First, train
        for _ in 0..num_iters {
            let mut iter_loss = Tensor::scalar_tensor(0f64, (Kind::Float, device));

            opt.zero_grad();
            
            let batch = training_examples.iter_training_batches(rng, device).collect();
            let iter_loss = get_loss_for_playout_bundles(&self, batch);

            let iter_loss_float = f64::from(&iter_loss);
            println!("Iter loss: {}", iter_loss_float);

            opt.backward_step(&iter_loss);
            total_train_loss += iter_loss_float / (num_iters as f64);
        }

        //Then, obtain validation loss
        let _guard = no_grad_guard();
        let mut validation_loss = Tensor::scalar_tensor(0f64, (Kind::Float, device)).detach();
        for validation_batch in training_examples.iter_validation_batches(device) {
            validation_loss += get_loss_for_playout_bundles(&self, validation_batch);
        }
        (total_train_loss, f64::from(validation_loss))
    }
}
