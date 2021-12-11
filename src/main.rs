#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod synthetic_data;
mod turn_data;
mod game_data;
mod game_tree;
mod normal_inverse_chi_squared;
mod array_utils;
mod matrix_set;
mod params;
mod neural_utils;
mod network;
mod network_config;
mod game_state;
mod training_examples;

use tch::{kind, Tensor, nn::Adam, nn::OptimizerConfig, Cuda};
use std::fs;
use std::env;
use crate::game_state::*;
use crate::params::*;
use crate::game_tree::*;
use crate::game_data::*;
use crate::training_examples::*;
use crate::network_config::*;
use std::str::from_utf8;
use std::path::Path;

fn print_help() {
    println!("usage:
ASMR run_game [game_config_path] [network_config_path] [game_data_output_path] [dotfile_output_path]?
    Using the given game configuration json and network configuration, randomly-generates
    and runs a game, outputting finalized game-data (training data) to the given output path.
    If [dotfile_output_path]? is present, this will also output a .dot file visualization
    of the game-tree once the simulation has completed.
ASMR gen_synthetic_training_data [game_config_path] [training_data_output_path]
    Using the given game configuration json, randomly-generates a bunch of synthetic
    games, outputting the finalized training data to the given output path
ASMR add_training_data [game_config_path] [training_data_to_add_path] [training_data_output_path]
    Using the given game configuration json, concatenates the training data
    at the given path to the training data at the output path, overwriting the destination
ASMR gen_network_config [game_config_path] [network_config_output_path]
    Using the given game configuration json, generates a randomly-initialized
    network configuration and outputs it to the given path
ASMR distill_game_data [game_config_path] [game_data_root] [training_data_output_path]
    Using the given game configuration json, distills all game data files under the
    given game data root directory into bona fide training data, which is output
    into a file at the given specified output path
ASMR train [game_config_path] [network_config_path] [training_data_path]
    Using the given game configuration json, trains the network
    with the initial configuration stored at the given network configuration path
    and the training data stored at the given training data path.
    Periodically [and also once training is completed], the updated
    state of the network will overwrite the data at the passed network configuration path.");
}

fn main() {
    let args : Vec<String> = env::args().collect();
    if (args.len() < 3) {
        print_help();
        return;
    }

    let command = &args[1];
    let game_config_path = &args[2];

    let maybe_game_config = load_game_config(game_config_path);
    match (maybe_game_config) {
        Result::Ok(params) => {
            match &command[..] {
                "add_training_data" => {
                    if (args.len() < 5) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_to_add_path = &args[3];
                    let training_data_output_path = &args[4];
                    add_training_data_command(params, training_data_to_add_path, training_data_output_path);
                },
                "gen_synthetic_training_data" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_output_path = &args[3];
                    gen_synthetic_training_data_command(params, training_data_output_path);
                }
                "distill_game_data" => {
                    if (args.len() < 5) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let game_data_root = &args[3];
                    let training_data_output_path = &args[4];
                    distill_game_data_command(params, game_data_root, training_data_output_path);
                },
                "run_game" => {
                    if (args.len() < 5) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let network_config_path = &args[3];
                    let training_data_output_path = &args[4];
                    let dotfile_output_path = if (args.len() > 5) {
                        Option::Some(args[5].clone())
                    } else {
                        Option::None
                    };
                    run_game_command(params, network_config_path, 
                                     training_data_output_path, dotfile_output_path);
                },
                "gen_network_config" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let network_config_output_path = &args[3];
                    gen_network_config_command(params, network_config_output_path);
                },
                "train" => {
                    if (args.len() < 5) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let network_config_path = &args[3];
                    let training_data_path = &args[4];
                    train_command(params, network_config_path, training_data_path);
                },
                _ => {
                    eprintln!("error: invalid command");
                    print_help();
                }
            }
        },
        Result::Err(err) => {
            eprintln!("Failed to read game config file: {}", err);
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

fn load_game_config(game_config_path : &str) -> Result<Params, String> {
    let maybe_game_config_contents = read_from_path(game_config_path);
    match (maybe_game_config_contents) {
        Result::Ok(path_contents) => {
            let maybe_param_json = std::str::from_utf8(&path_contents);
            match (maybe_param_json) {
                Result::Ok(param_json) => {
                    let maybe_params = serde_json::from_str::<Params>(param_json);
                    match (maybe_params) {
                        Result::Ok(params) => {
                            Result::Ok(params)
                        },
                        Result::Err(err) => {
                            Result::Err(format!("Game config deserialization error: {}", err))
                        }
                    }
                },
                Result::Err(err) => {
                    Result::Err(format!("Game config json error: {}", err))
                }
            }
        },
        Result::Err(err) => {
            Result::Err(format!("Failed to read game config file: {}", err))
        }
    }
}
fn gen_network_config_command(params : Params, network_config_output_path : &str) {
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let _network_config = NetworkConfig::new(&params, &vs.root());
    let path = Path::new(network_config_output_path);
    let maybe_save_result = vs.save(&path);
    match (maybe_save_result) {
        Result::Ok(_) => {
            println!("Successfully wrote out a new network configuration");
        },
        Result::Err(e) => {
            eprintln!("Failed to write out network configuration: {}", e);
        }
    }
}

fn run_game_command(params : Params, 
                    network_config_path : &str,
                    game_data_output_path : &str,
                    maybe_dotfile_output_path : Option<String>) {
    let mut rng = rand::thread_rng();
    let game_state = params.generate_random_game(&mut rng);
    let mut game_tree = GameTree::new(game_state);

    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let network_config = NetworkConfig::new(&params, &vs.root());

    let maybe_load_result = vs.load(&Path::new(network_config_path));
    match (maybe_load_result) {
        Result::Ok(_) => {
            println!("Successfully loaded network config");
        },
        Result::Err(e) => {
            eprintln!("Failed to read network config: {}", e);
            return;
        }
    }

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
    let game_data = game_tree.extract_game_data();
    let maybe_serialized_game_data = bincode::serialize(&game_data);
    match (maybe_serialized_game_data) {
        Result::Ok(serialized_game_data) => {
            let maybe_write_result = write_to_path(game_data_output_path, &serialized_game_data);
            match (maybe_write_result) {
                Result::Ok(_) => {
                    println!("Successfully wrote out generated game data");
                },
                Result::Err(err) => {
                    println!("Failed to write out game data: {}", err);
                }
            }
        },
        Result::Err(err) => {
            println!("Game data serialization error: {}", err);
        }
    }
}

fn add_training_data_command(_params : Params, training_data_to_add_path : &str, training_data_output_path : &str) {
    let to_add_path = Path::new(training_data_to_add_path);
    let output_path = Path::new(training_data_output_path);

    let maybe_training_examples_to_add = TrainingExamples::load(&to_add_path, tch::Device::Cpu);
    let training_examples_to_add = match (maybe_training_examples_to_add) {
        Result::Ok(x) => x,
        Result::Err(err) => {
            eprintln!("Failed to load training examples to add: {}", err);
            return;
        }
    };

    let maybe_training_examples_to_add_to = TrainingExamples::load(&output_path, tch::Device::Cpu);
    let mut training_examples_to_add_to = match (maybe_training_examples_to_add_to) {
        Result::Ok(x) => x,
        Result::Err(err) => {
            eprintln!("Failed to load training examples to add to: {}", err);
            return;
        }
    };

    training_examples_to_add_to.merge(training_examples_to_add);
    let maybe_save_result = training_examples_to_add_to.save(&output_path); 
    match (maybe_save_result) {
        Result::Ok(_) => {
            println!("Successfully added and saved training data");
        },
        Result::Err(err) => {
            eprintln!("Failed to save combined training data: {}", err);
        }
    }
}

fn gen_synthetic_training_data_command(params : Params, training_data_output_path : &str) {
    let mut rng = rand::thread_rng();
    let mut builder = TrainingExamplesBuilder::new(&params);

    for _ in 0..params.num_synthetic_training_games {
        let game_path = params.generate_random_game_path(&mut rng);
        let game_data = game_path.get_game_data();

        builder.add_game_data(game_data, &mut rng); 
    }
    let training_examples = builder.build(&mut rng);

    let output_path = Path::new(training_data_output_path);
    let maybe_save_result = training_examples.save(&output_path);
    match (maybe_save_result) {
        Result::Ok(_) => {
            println!("Successfully generated and saved synthetic training data");
        },
        Result::Err(err) => {
            eprintln!("Failed to write synthetic training data: {}", err);
        }
    }
}

fn distill_game_data_command(params : Params, game_data_root : &str, training_data_output_path : &str) {
    let game_data_root_path = Path::new(game_data_root);
    let maybe_dir_listing = fs::read_dir(&game_data_root_path);

    let mut rng = rand::thread_rng();
    let mut builder = TrainingExamplesBuilder::new(&params);

    match (maybe_dir_listing) {
        Result::Ok(dir_listing) => {
            for maybe_entry in dir_listing {
                match (maybe_entry) {
                    Result::Ok(entry) => {
                        let entry_path = entry.path();
                        let maybe_serialized_game_data = read_from_path(entry_path.to_str().unwrap());
                        match (maybe_serialized_game_data) {
                            Result::Ok(serialized_game_data) => {
                                let maybe_game_data = bincode::deserialize::<GameData>(&serialized_game_data);
                                match (maybe_game_data) {
                                    Result::Ok(game_data) => {
                                        builder.add_game_data(game_data, &mut rng);
                                    },
                                    Result::Err(err) => {
                                        eprintln!("Failed to deserialize contents of a file in the directory: {}", err);
                                        return;
                                    }
                                }
                            },
                            Result::Err(err) => {
                                eprintln!("Failed to read contents of a file in the directory: {}", err);
                                return;
                            }
                        }
                    },
                    Result::Err(err) => {
                        eprintln!("Failed to access contents of a file in the directory: {}", err);
                        return;
                    }
                }
            }
        },
        Result::Err(err) => {
            eprintln!("Failed to list game data directory contents: {}", err);
            return;
        }
    }
    let training_examples = builder.build(&mut rng);

    let output_path = Path::new(training_data_output_path);
    let maybe_save_result = training_examples.save(&output_path);
    match (maybe_save_result) {
        Result::Ok(_) => {
            println!("Successfully distilled and saved training data");
        },
        Result::Err(err) => {
            eprintln!("Failed to write distilled training data: {}", err);
        }
    }
}

fn train_command(params : Params, network_config_path : &str, training_data_path : &str) {
    let device = tch::Device::Cuda(params.gpu_slot);
    println!("Is cuDNN available?: {}", Cuda::cudnn_is_available());
    Cuda::set_user_enabled_cudnn(true);
    Cuda::cudnn_set_benchmark(true);
    let mut vs = tch::nn::VarStore::new(device);
    let network_config = NetworkConfig::new(&params, &vs.root());

    let network_path = Path::new(network_config_path);

    let maybe_load_result = vs.load(&network_path);
    match (maybe_load_result) {
        Result::Ok(_) => {
            println!("Successfully loaded initial network config");
        },
        Result::Err(e) => {
            eprintln!("Failed to read initial network config: {}", e);
            return;
        }
    }

    let mut rng = rand::thread_rng();
    let data_path = Path::new(training_data_path);
    let maybe_training_examples = TrainingExamples::load(&data_path, tch::Device::Cpu);
    match (maybe_training_examples) {
        Result::Ok(training_examples) => { 
            println!("Successfully loaded training examples");

            let adam = Adam {
                wd : params.weight_decay_factor,
                ..Adam::default()
            };
            let mut opt = adam.build(&vs, params.train_step_size).unwrap();
            for epoch in 0..params.train_epochs {
                let train_loss = network_config.run_training_epoch(&params, &training_examples, 
                                                                   &mut opt, device, &mut rng);
                println!("epoch: {} train loss: {}", epoch, train_loss);

                //Write out the updated network weights
                let maybe_save_result = vs.save(&network_path);
                match (maybe_save_result) {
                    Result::Ok(_) => {
                        println!("Successfully wrote out the updated network configuration");
                    },
                    Result::Err(e) => {
                        eprintln!("Failed to write out updated network configuration: {}", e);
                    }
                }
            }
        },
        Result::Err(err) => {
            eprintln!("Failed to read training examples: {}", err);
            return;
        }
    }
}
