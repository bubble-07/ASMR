use crate::game_state::*;
use crate::matrix_set::*;
use rand::Rng;
use rand_distr::{Uniform, Geometric, LogNormal, Distribution};
use ndarray::*;
use ndarray_rand::RandomExt;
use ndarray::{Array, ArrayBase};
use std::convert::TryInto;
use serde::{Serialize, Deserialize};
use crate::synthetic_data::*;

#[derive(Serialize, Deserialize)]
pub struct Params {
    ///Dimension of the vector space the matrices operate on
    pub matrix_dim : usize,
    ///Number of feature maps for both the singleton-injection and combining nets
    pub num_feat_maps : i64,
    ///Number of layers for the singleton injection net
    pub singleton_injection_layers : usize,
    ///Number of layers for the combining net
    pub combining_layers : usize,
    ///Number of layers for the policy extraction net
    pub policy_extraction_layers : usize,
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
    ///Number of monte-carlo-tree-search updates per game
    pub iters_per_game : usize,
    ///Number of epochs for training
    pub train_epochs : usize,
    ///Step-size for Adam optimizer
    pub train_step_size : f64,
    ///Number of synthetic training games
    pub num_synthetic_training_games : usize
}

impl Params {
    pub fn get_flattened_matrix_dim(&self) -> i64 {
        (self.matrix_dim * self.matrix_dim).try_into().unwrap()
    }

    fn generate_matrix_set<R : Rng + ?Sized>(&self, set_size : usize, rng : &mut R) -> MatrixSet {
        let log_normal = LogNormal::new(0.0, self.log_normal_std_dev).unwrap();

        let mut matrices = Vec::new();

        for _ in 0..set_size {
            let uniform_zero_one = Uniform::new_inclusive(0, 1);
            let sign_distribution = uniform_zero_one.map(|num| (num as f64) * 2.0f64 - 1.0f64);

            let unsigned_matrix = Array::random_using((self.matrix_dim, self.matrix_dim), log_normal, rng);
            let sign_matrix = Array::random_using((self.matrix_dim, self.matrix_dim), sign_distribution, rng);
            let wide_matrix = &sign_matrix * &unsigned_matrix;
            let matrix = wide_matrix.mapv(|x| x as f32);

            matrices.push(matrix);
        }

        MatrixSet {
            matrices
        }
    }

    fn generate_target_matrix<R : Rng + ?Sized>(matrix_set : &MatrixSet, 
                                                ground_truth_num_moves : usize, 
                                                rng : &mut R) -> Array2<f32> {
        let matrix_set_clone = matrix_set.clone();
        let game_path = GamePath::generate_game_path(matrix_set_clone, ground_truth_num_moves, rng);
        game_path.get_target()
    }

    fn generate_initial_set_size<R : Rng + ?Sized>(&self, rng : &mut R) -> usize {
        rng.gen_range(self.min_initial_set_size..=self.max_initial_set_size)
    }

    fn generate_ground_truth_num_moves<R : Rng + ?Sized>(&self, rng : &mut R) -> usize {
        let geom_distr = Geometric::new(self.success_probability_ground_truth_num_moves).unwrap();
        let additional_moves_beyond_one = geom_distr.sample(rng) as usize;
        additional_moves_beyond_one + 1
    }

    pub fn generate_random_game_path<R : Rng + ?Sized>(&self, rng : &mut R) -> GamePath {
        let initial_set_size = self.generate_initial_set_size(rng);
        let matrix_set = self.generate_matrix_set(initial_set_size, rng);
        let ground_truth_num_moves = self.generate_ground_truth_num_moves(rng);
        
        GamePath::generate_game_path(matrix_set, ground_truth_num_moves, rng)
    }

    pub fn generate_random_game<R : Rng + ?Sized>(&self, rng : &mut R) -> GameState {
        let initial_set_size = self.generate_initial_set_size(rng);
        let ground_truth_num_moves = self.generate_ground_truth_num_moves(rng);

        let remaining_turns = self.max_num_turns;
        let matrix_set = self.generate_matrix_set(initial_set_size, rng);

        let target = Self::generate_target_matrix(&matrix_set, ground_truth_num_moves, rng);

        GameState::new(matrix_set, target, remaining_turns)
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
