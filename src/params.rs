use std::rc::Rc;
use crate::game_tree_trait::*;
use crate::network_config::*;
use crate::random_game_tree::*;
use crate::network_game_tree::*;
use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, kind::Kind,
          nn::Optimizer, IndexOp, Device};
use rand::Rng;
use rand_distr::{Uniform, Geometric, LogNormal, Distribution, StandardNormal};
use ndarray::*;
use ndarray::{Array, ArrayBase};
use std::convert::TryInto;
use serde::{Serialize, Deserialize};
use crate::synthetic_data::*;
use crate::rollout_states::*;
use crate::training_examples::*;
use crate::playout_sketches::*;
use crate::generate;

#[derive(Serialize, Deserialize)]
pub struct Params {
    ///Dimension of the vector space the matrices operate on
    pub matrix_dim : usize,
    ///Number of feature maps used in all intermediate layers in the network
    pub num_feat_maps : usize,
    ///Number of layers for the main net with attention layers
    pub num_main_net_layers : usize,
    ///Number of layers for extracting the policy from (single, single, global) triples
    pub num_policy_extraction_layers : usize,
    ///Standard deviation in log-space for generating random matrix entries
    pub log_normal_std_dev : f64,
    ///Minimum number of initial matrices to place in the randomly-generated set
    pub min_initial_set_size : usize,
    ///Maximum number of initial matrices to place in the randomly-generated set
    pub max_initial_set_size : usize,
    ///Success probability for geometric distribution which generates the number of moves
    ///_actually_ required (>= 1) to reach the target matrix
    pub success_probability_ground_truth_num_moves : f64,
    ///Upper-bound on the number of turns allowed in the game
    pub max_num_turns : usize,
    ///Cap on the number of monte-carlo-tree-search updates per game.
    ///In the case of "run_game", exactly this many iters will be performed,
    ///but in the case of "time_games", this is the max number of iters per game,
    ///since achieving a distance of zero to the target ends the timing episode.
    pub iters_per_game : usize,
    ///Number of games for timing purposes
    pub num_timing_games : usize,
    ///The strategy to employ for performing rollouts during MCTS.
    ///Can be "random", or "network"
    pub rollout_strategy : String,
    ///Number of batches before evaluating validation loss
    ///and potentially saving out the updated network configuration for training
    pub train_batches_per_save : usize,
    ///Step-size for Adam optimizer
    pub train_step_size : f64,
    ///Number of synthetic training games per synthetic training data file
    pub num_synthetic_training_games : usize,
    ///Number of synthetic training data files to generate
    pub num_synthetic_training_data_files : usize,
    ///Weight decay factor for training. (Helps regularize the model)
    pub weight_decay_factor : f64,
    ///Batch size for training
    pub batch_size : usize,
    ///Number of batches to hold out of the training set for validation
    pub held_out_validation_batches : usize,
    ///Whether or not to freeze all layers other than the injector layers.
    ///Useful when expanding the input size of a model
    pub freeze_non_injector_layers : bool,
    ///GPU slot to use for training
    pub gpu_slot : usize
}

pub enum RolloutStrategy {
    Random,
    NetworkConfig,
}

impl RolloutStrategy {
    pub fn build_game_tree(&self, network_config : Rc<NetworkConfig>, 
                           game_state : RolloutStates) -> Box<dyn GameTreeTraverserTrait> {
        match self {
            RolloutStrategy::Random => {
                Box::new(RandomTreeTraverser::build_from_game_state(game_state))
            },
            RolloutStrategy::NetworkConfig => {
                Box::new(NetworkTreeTraverser::build_from_game_state(network_config, game_state))
            },
        }
    }
}

impl Params {
    pub fn get_device(&self) -> Device {
        Device::Cuda(self.gpu_slot)
    }
    pub fn get_rollout_strategy(&self) -> RolloutStrategy {
        match (self.rollout_strategy.as_str()) {
            "random" => RolloutStrategy::Random,
            "network" => RolloutStrategy::NetworkConfig,
            x => {
                eprintln!("Failed to interpret rollout strategy: {}", x);
                panic!();
            }
        }
    }
    pub fn get_flattened_matrix_dim(&self) -> i64 {
        (self.matrix_dim * self.matrix_dim).try_into().unwrap()
    }

    ///Dims N x M x M
    pub fn generate_random_matrices(&self, N : usize) -> Tensor {
        generate::random_log_normal_singular_value_matrices(N, self.matrix_dim, self.log_normal_std_dev, self.get_device())
    }

    pub fn generate_random_playout<R : Rng + ?Sized>(&self, rng : &mut R) -> PlayoutBundle {
        let annotated_game_path = self.generate_random_game_path(rng);
        let playout_sketch_bundle = PlayoutSketchBundle::from_single_annotated_game_path(annotated_game_path); 
        let playout_bundle = PlayoutBundle::from_sketch_bundle(&self, playout_sketch_bundle);
        playout_bundle
    }

    pub fn generate_random_standard_game<R : Rng + ?Sized>(&self, rng : &mut R) -> RolloutStates {
        let random_playout = self.generate_random_playout(rng);
        let standard_playout = random_playout.standardize();
        RolloutStates::from_playout_bundle_initial_state(&standard_playout)
    }

    fn generate_initial_set_size<R : Rng + ?Sized>(&self, rng : &mut R) -> usize {
        rng.gen_range(self.min_initial_set_size..=self.max_initial_set_size)
    }

    fn generate_ground_truth_num_moves<R : Rng + ?Sized>(&self, rng : &mut R) -> usize {
        let geom_distr = Geometric::new(self.success_probability_ground_truth_num_moves).unwrap();
        //We omit single-move games, because those may be found straightforwardly by brute-force
        let additional_moves_beyond_two = geom_distr.sample(rng) as usize;
        additional_moves_beyond_two + 2
    }

    pub fn generate_random_game_path<R : Rng + ?Sized>(&self, rng : &mut R) -> AnnotatedGamePath {
        let initial_set_size = self.generate_initial_set_size(rng);
        let ground_truth_num_moves = self.generate_ground_truth_num_moves(rng);
        
        AnnotatedGamePath::generate_game_path(initial_set_size, ground_truth_num_moves, rng)
    }
}
//Random notes: Rubik's cube group, for instance, can have representations with matrix dimension ~20,
//and ~6 generators. 20 turns ["God's number"] is the upper bound on the minimal number of terms.
//This would be useful for a particularly inspired set of choices of parameters,
//and from this, we could also potentially _measure_ performance on solving Rubik's cubes
//as a benchmark [bearing in mind that we're solving a different, more general problem]
//
//For the number of moves, though, it's important to bear in mind that our "moves"
//are multiplication of _two_ matrices. If we take the naive assumption that
//repeated squarings would suffice, then our reference number of moves is
//log_2(20) which is about 4.3219.
//So, setting the half-life of a geometric distribution for # moves to happen around 4 seems reasonable
