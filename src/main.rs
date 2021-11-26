#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod training_example;
mod game_tree;
mod normal_inverse_chi_squared;
mod array_utils;
mod matrix_set;
mod params;
mod neural_utils;
mod network;
mod network_config;
mod game_state;

use tch::{kind, Tensor};
use std::fs;
use std::env;
use crate::game_state::*;
use crate::params::*;
use crate::game_tree::*;
use crate::training_example::*;
use crate::network_config::*;
use std::str::from_utf8;

fn print_help() {
    println!("usage:
ASMR run_game [game_config_path] [network_config_path] [training_data_output_path] [dotfile_output_path]?
    Using the given game configuration json and network configuration, randomly-generates
    and runs a game, outputting finalized training-data to the given output path.
    If [dotfile_output_path]? is present, this will also output a .dot file visualization
    of the game-tree once the simulation has completed");
}

fn main() {
    let args : Vec<String> = env::args().collect();
    if (args.len() < 2) {
        print_help();
        return;
    }

    let command = &args[1];
    match &command[..] {
        "run_game" => {
            if (args.len() < 5) {
                eprintln!("error: not enough arguments");
                print_help();
                return;
            }
            let game_config_path = &args[2];
            let network_config_path = &args[3];
            let training_data_output_path = &args[4];
            let dotfile_output_path = if (args.len() > 5) {
                Option::Some(args[5].clone())
            } else {
                Option::None
            };
            run_game_command(game_config_path, network_config_path, 
                             training_data_output_path, dotfile_output_path);
        },
        _ => {
            eprintln!("error: invalid command");
            print_help();
        }
    }
}


pub fn read_from_path(path : &str) -> Result<Vec<u8>, String> {
    let maybe_canonical_path = shellexpand::full(path);
    match (maybe_canonical_path) {
        Result::Ok(canonical_path) => {
            let maybe_path_contents = fs::read(&*canonical_path);
            match (maybe_path_contents) {
                Result::Ok(path_contents) => Result::Ok(path_contents),
                Result::Err(err) => Result::Err(format!("Read Error: {}", err))
            }
        },
        Result::Err(err) => Result::Err(format!("Path Resolution Error: {}", err))
    }
}

pub fn write_to_path(path : &str, contents : &[u8]) -> Result<(), String> {
    let maybe_canonical_path = shellexpand::full(path);
    match (maybe_canonical_path) {
        Result::Ok(canonical_path) => {
            let maybe_write_result = fs::write(&*canonical_path, contents);
            match (maybe_write_result) {
                Result::Ok(_) => Result::Ok(()),
                Result::Err(err) => Result::Err(format!("Writing Error: {}", err))
            }
        },
        Result::Err(err) => Result::Err(format!("Path Resolution Error: {}", err))
    }
}



fn run_game_command(game_config_path : &str, network_config_path : &str, 
            training_data_output_path : &str, dotfile_output_path : Option<String>) {

    let maybe_game_config_contents = read_from_path(game_config_path);
    match (maybe_game_config_contents) {
        Result::Ok(path_contents) => {
            let maybe_param_json = std::str::from_utf8(&path_contents);
            match (maybe_param_json) {
                Result::Ok(param_json) => {
                    let maybe_params = serde_json::from_str::<Params>(param_json);
                    match (maybe_params) {
                        Result::Ok(params) => {
                            run_game(params, training_data_output_path, dotfile_output_path);
                        },
                        Result::Err(err) => {
                            eprintln!("Game config deserialization error: {}", err);
                        }
                    }
                },
                Result::Err(err) => {
                    eprintln!("Game config json error: {}", err);
                }
            }
        },
        Result::Err(err) => {
            eprintln!("Failed to read game config: {}", err);
        }
    }
}

fn run_game(params : Params, training_data_output_path : &str,
                    maybe_dotfile_output_path : Option<String>) {
    let mut rng = rand::thread_rng();
    let game_state = params.generate_random_game(&mut rng);
    let mut game_tree = GameTree::new(game_state);

    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let network_config = NetworkConfig::new(&params, &vs.root());

    for i in 0..params.iters_per_game {
        println!("Iteration: {}", i);
        game_tree.update_iteration(&network_config, &mut rng);
    }
    match (maybe_dotfile_output_path) {
        Option::Some(dotfile_output_path) => {
            let dotfile_contents = game_tree.render_dotfile();
            let write_result = write_to_path(&dotfile_output_path, &dotfile_contents.as_bytes());
            match (write_result) {
                Result::Ok(_) => {
                    println!("Successfully wrote out dotfile");
                },
                Result::Err(err) => {
                    println!("Failed to write out dotfile: {}", err);
                }
            }
        },
        _ => {}
    }
    let training_data = game_tree.extract_training_examples();
    let maybe_serialized_training_data = bincode::serialize(&training_data);
    match (maybe_serialized_training_data) {
        Result::Ok(serialized_training_data) => {
            let maybe_write_result = write_to_path(training_data_output_path, &serialized_training_data);
            match (maybe_write_result) {
                Result::Ok(_) => {
                    println!("Successfully wrote out generated training data");
                },
                Result::Err(err) => {
                    println!("Failed to write out training data: {}", err);
                }
            }
        },
        Result::Err(err) => {
            println!("Training data serialization error: {}", err);
        }
    }
}
