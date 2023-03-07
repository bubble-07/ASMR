//#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_parens)]

mod matrix_bundle;
mod validation_set;
mod playout_sketches;
mod network_game_tree;
mod random_game_tree;
mod game_tree_trait;
mod tree;
mod tweakable_tensor;
mod peeling_states;
mod bunched_rollout;
mod visit_logit_matrices;
mod network_module;
mod synthetic_data;
mod rollout_states;
mod normal_inverse_chi_squared;
mod array_utils;
mod params;
mod neural_utils;
mod network;
mod network_config;
mod batch_split_training_examples;
mod training_examples;
mod network_rollout;
mod batch_split;

use tch::{kind, Tensor, nn::Adam, nn::OptimizerConfig, Cuda, IndexOp};
use std::fs;
use std::rc::Rc;
use std::sync::mpsc::*;
use std::cmp::min;
use crate::game_tree_trait::DynRng;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::env;
use crate::params::*;
use crate::training_examples::*;
use crate::playout_sketches::*;
use crate::validation_set::*;
use crate::synthetic_data::*;
use crate::network_config::*;
use crate::network::*;
use crate::batch_split_training_examples::*;
use std::str::from_utf8;
use std::path::Path;
use rayon::*;
use rayon::iter::*;

fn print_help() {
    println!("usage:
ASMR time_games [game_config_path] [network_config_path] [game_time_csv_path]
    Using the given game configuration json and network configuration, randomly-generates
    and runs a bunch of games, outputting the time since game start -> minimal distance from target
    distribution for each game, one after the other.
ASMR run_game [game_config_path] [network_config_path] [game_data_output_path] [dotfile_output_path]?
    Using the given game configuration json and network configuration, randomly-generates
    and runs a game, outputting finalized game-data (training data) to the given output path.
    If [dotfile_output_path]? is present, this will also output a .dot file visualization
    of the game-tree once the simulation has completed.
ASMR gen_synthetic_training_data_files [game_config_path] [training_data_output_dir]
    Using the given game configuration json, randomly-generates a bunch of synthetic
    games, outputting the results to a directory filled with finalized
    training data files with sequential names data0.tensor, data1.tensor, ... etc.
ASMR gen_synthetic_training_data_file [game_config_path] [training_data_output_path]
    Using the given game configuration json, randomly-generates a bunch of synthetic
    games, outputting the finalized training data file to the given output path
ASMR add_training_data [game_config_path] [training_data_to_add_path] [training_data_output_path]
    Using the given game configuration json, concatenates the training data
    at the given path to the training data at the output path, overwriting the destination
ASMR display_training_data_stats [game_config_path] [training_data_input_path]
    Using the given game configuration json, displays a bunch of statistics for the
    training data loaded from the given input path.
ASMR gen_network_config [game_config_path] [network_config_output_path]
    Using the given game configuration json, generates a randomly-initialized
    network configuration and outputs it to the given path
ASMR distill_game_data [game_config_path] [game_data_root] [training_data_output_path]
    Using the given game configuration json, distills all game data files under the
    given game data root directory into bona fide training data, which is output
    into a file at the given specified output path
ASMR recast_network [old_game_config_path] [new_game_config_path] 
                    [network_config_input_path] [network_config_output_path]
    Given an old and a new game configuration and an old network with pre-trained weights,
    carries over as much from the previous network configuration as possible,
    leaving any undetermined values randomized. This currently only works for changing
    the size of the input matrices.
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
                "display_training_data_stats" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_input_path = &args[3];
                    display_training_data_stats_command(params, training_data_input_path);
                },
                "add_training_data" => {
                    if (args.len() < 5) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_to_add_path = &args[3];
                    let training_data_output_path = &args[4];
                    add_training_data_command::<PlayoutBundle>(params, training_data_to_add_path, training_data_output_path);
                },
                "gen_synthetic_training_data_file" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_output_path = &args[3];
                    gen_synthetic_training_data_file_command(&params, training_data_output_path);
                },
                "gen_synthetic_training_data_files" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let training_data_output_dir = &args[3];
                    gen_synthetic_training_data_files_command(params, training_data_output_dir);
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
                "time_games" => {
                    if (args.len() < 4) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let network_config_path = &args[3];
                    let timing_csv_output_path = &args[4];
                    time_games_command(params, network_config_path, timing_csv_output_path); 
                }
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
                "recast_network" => {
                    if (args.len() < 6) {
                        eprintln!("error: not enough arguments");
                        print_help();
                        return;
                    }
                    let old_params = params;

                    let new_network_params_path = &args[3];
                    let maybe_new_config = load_game_config(new_network_params_path);

                    match (maybe_new_config) {
                        Result::Ok(new_params) => {
                            let network_config_input_path = &args[4];                    
                            let network_config_output_path = &args[5];
                            recast_command(old_params, new_params,
                                           network_config_input_path, network_config_output_path);
                        },
                        Result::Err(err) => {
                            eprintln!("Failed to read new game config file: {}", err);
                        },
                    }
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

fn recast_command(old_params : Params, new_params : Params,
                  network_config_input_path : &str, 
                  network_config_output_path : &str) {
    let mut old_vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let old_network_config = NetworkConfig::new(&old_params, &old_vs.root());

    let maybe_load_result = old_vs.load(&Path::new(network_config_input_path));
    match (maybe_load_result) {
        Result::Ok(_) => {
            println!("Successfully loaded network config, recasting");
        },
        Result::Err(e) => {
            eprintln!("Failed to read network config: {}", e);
            panic!();
        }
    }
    old_vs.freeze();

    //Generate a new network using the new config
    let mut new_vs = tch::nn::VarStore::new(tch::Device::Cpu);
    let new_network_config = NetworkConfig::new(&new_params, &new_vs.root());

    new_vs.freeze();

    //Copy over all the old tensor values which are not part of the injector network
    let mut old_variables = old_vs.variables();
    let mut new_variables = new_vs.variables();
    for (var_name, old_tensor) in old_variables.drain() {
        let new_tensor = new_variables.get_mut(&var_name).unwrap();
        if !var_name.contains("injector") {
            new_tensor.copy_(&old_tensor);
        } else if var_name.contains("weight") {
            //Injector network's weights
            let old_matrix_dim = old_params.matrix_dim as i64;
            let new_matrix_dim = new_params.matrix_dim as i64;
            let min_matrix_dim = min(old_matrix_dim, new_matrix_dim);

            let old_matrix_size = old_matrix_dim * old_matrix_dim;
            let new_matrix_size = new_matrix_dim * new_matrix_dim;

            let reshaped_old_tensor = 
                old_tensor.reshape(&[old_params.num_feat_maps as i64, old_matrix_dim, old_matrix_dim]);
            let reshaped_new_tensor =
                new_tensor.reshape(&[new_params.num_feat_maps as i64, new_matrix_dim, new_matrix_dim]);
        
            reshaped_new_tensor.i((.., ..min_matrix_dim, ..min_matrix_dim))
                               .copy_(&reshaped_old_tensor);

            let reshaped_new_tensor = reshaped_new_tensor.reshape(&[new_params.num_feat_maps as i64,
                                                                    new_matrix_size]);
            new_tensor.copy_(&reshaped_new_tensor);
        } else if var_name.contains("bias") {
            //Injector bias, copy this over, since it should have the same dims
            new_tensor.copy_(&old_tensor);
        }
    }

    new_vs.unfreeze();

    //Output the updated network
    let maybe_save_result = new_vs.save(&Path::new(network_config_output_path));
    match (maybe_save_result) {
        Result::Ok(_) => {
            println!("Successfully wrote out a new network configuration");
        },
        Result::Err(e) => {
            eprintln!("Failed to write out network configuration: {}", e);
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

fn load_network_config_for_inference(params : &Params, network_config_path : &str) -> NetworkConfig {
    let device = tch::Device::Cuda(params.gpu_slot);
    let mut vs = tch::nn::VarStore::new(device);
    let network_config = NetworkConfig::new(params, &vs.root());

    let maybe_load_result = vs.load(&Path::new(network_config_path));
    match (maybe_load_result) {
        Result::Ok(_) => {
            println!("Successfully loaded network config");
            network_config
        },
        Result::Err(e) => {
            eprintln!("Failed to read network config: {}", e);
            panic!();
        }
    }
}

fn time_games_command(params : Params,
                      network_config_path : &str,
                      timing_csv_output_path : &str) {
    let mut rng = rand::thread_rng();
    let mut rng = DynRng::from(&mut rng);

    let device = tch::Device::Cuda(params.gpu_slot);

    let network_config = load_network_config_for_inference(&params, network_config_path);
    let network_config = Rc::new(network_config);
    let rollout_strategy = params.get_rollout_strategy();

    for game_number in 0..params.num_timing_games {
        let game_state = params.generate_random_standard_game(&mut rng);
        let mut min_distance : f32 = f32::from(&game_state.min_distances);

        let mut game_tree = rollout_strategy.build_game_tree(network_config.clone(), game_state);

        let mut observations = Vec::new();
        
        observations.push((0, 0, min_distance));

        println!("Timing game {}...", game_number);
        println!("{}, {}, {}", 0, 0, min_distance);

        let start_instant = Instant::now();
        let mut reached_goal = false;
        
        for iter_number in 0..params.iters_per_game {
            let distance = game_tree.perform_update_iteration(&mut rng);
            if (distance < min_distance) {
                min_distance = distance;
                let duration_elapsed = start_instant.elapsed(); 
                let value_elapsed = duration_elapsed.as_millis();
                let observation = (iter_number + 1, value_elapsed, distance);
                observations.push(observation);
                println!("{}, {}, {}", iter_number + 1, value_elapsed, distance);

                if (distance < 0.0001f32) {
                    reached_goal = true;
                    //If we reach the goal, we're done with this game, move on to the next.
                    break;
                }
            }
        }
        if (!reached_goal) {
            println!("Hit iter cap of {}", params.iters_per_game);
        }
    }

}


fn run_game_command(params : Params, 
                    network_config_path : &str,
                    game_data_output_path : &str,
                    maybe_dotfile_output_path : Option<String>) {
    let mut rng = rand::thread_rng();
    let mut rng = DynRng::from(&mut rng);
    let game_state = params.generate_random_standard_game(&mut rng);
    let device = tch::Device::Cuda(params.gpu_slot);

    let network_config = load_network_config_for_inference(&params, network_config_path);
    let network_config = Rc::new(network_config);
    let rollout_strategy = params.get_rollout_strategy();

    let mut game_tree = rollout_strategy.build_game_tree(network_config, game_state);

    for i in 0..params.iters_per_game {
        println!("Iteration: {}", i);
        game_tree.perform_update_iteration(&mut rng);
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
    //TODO: reimplement game-tree serialization
    //let maybe_serialized_game_data = bincode::serialize(&game_tree);
    //match (maybe_serialized_game_data) {
    //    Result::Ok(serialized_game_data) => {
    //        let maybe_write_result = write_to_path(game_data_output_path, &serialized_game_data);
    //        match (maybe_write_result) {
    //            Result::Ok(_) => {
    //                println!("Successfully wrote out generated game data");
    //            },
    //            Result::Err(err) => {
    //                println!("Failed to write out game data: {}", err);
    //            }
    //        }
    //    },
    //    Result::Err(err) => {
    //        println!("Game data serialization error: {}", err);
    //    }
    //}
}

fn display_training_data_stats_command(_params : Params, training_data_input_path : &str) {
    let input_path = Path::new(training_data_input_path);

    let maybe_training_examples = TrainingExamples::load(&input_path, tch::Device::Cpu);

    //TODO: May want to generalize the stats thing to the sketch playout bundles, too
    let training_examples : TrainingExamples<PlayoutBundle> = match (maybe_training_examples) {
        Result::Ok(x) => x,
        Result::Err(err) => {
            eprintln!("Failed to load training examples: {}", err);
            return;
        }
    };

    println!("Stats for playout bundles:");
    let mut number_and_strings = Vec::new();
    for ((init_set_size, playout_length), playout_bundle) in training_examples.playout_bundles {
        let num_examples = playout_bundle.get_num_playouts();
        let mut string = format!("N: {}, Init Set Size: {}, Playout length: {}\n", 
                                 num_examples, init_set_size, playout_length);
        let matrix_target_elements : Vec<f32> = playout_bundle.matrix_bundle.flattened_matrix_targets.into();
        let std_dev = statistical::standard_deviation(&matrix_target_elements, None);
        string += format!("\tTarget matrix std_dev: {}\n", std_dev).as_str();
        let final_set_size = init_set_size + playout_length;
        let final_matrix_size = final_set_size * final_set_size;
        let heaviness = ((num_examples * final_matrix_size) as f64) * (std_dev as f64) / 10000.0;
        string += format!("\tHeaviness: {}", heaviness).as_str();
        number_and_strings.push((num_examples, string));
    }
    number_and_strings.sort_by_key(|(n, k)| *n);
    number_and_strings.reverse();
    
    for (_, string) in number_and_strings.drain(..) {
        println!("{}", string);
    }
}

fn add_training_data_command<BundleType : PlayoutBundleLike>(_params : Params, 
                                                             training_data_to_add_path : &str, training_data_output_path : &str) {
    let to_add_path = Path::new(training_data_to_add_path);
    let output_path = Path::new(training_data_output_path);

    let maybe_training_examples_to_add = TrainingExamples::load(&to_add_path, tch::Device::Cpu);
    let training_examples_to_add : TrainingExamples<BundleType> = match (maybe_training_examples_to_add) {
        Result::Ok(x) => x,
        Result::Err(err) => {
            eprintln!("Failed to load training examples to add: {}", err);
            return;
        }
    };

    let maybe_training_examples_to_add_to = TrainingExamples::load(&output_path, tch::Device::Cpu);
    let mut training_examples_to_add_to : TrainingExamples<BundleType> = match (maybe_training_examples_to_add_to) {
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

fn gen_synthetic_training_data_files_command(params : Params, training_data_output_dir : &str) {
    for i in 0..params.num_synthetic_training_data_files {
        let filename = format!("{}/data{}.tensor", training_data_output_dir, i);
        gen_synthetic_training_data_file_command(&params, &filename);
    }
}

fn gen_synthetic_training_data_file_command(params : &Params, training_data_output_path : &str) {
    //A bunch of worker threads will send AnnotatedGamePaths
    //to a single writer thread which is responsible for
    //collating responses into a training-examples builder.
    let (sender, receiver) = channel(); 

    let builder = SketchExamplesBuilder::new();
    let builder_thread_handle = std::thread::spawn(move || {
        let mut builder = builder;
        loop {
            let maybe_game_path = receiver.recv();
            match (maybe_game_path) {
                Result::Ok(Option::Some(game_path)) => {
                    builder.add_annotated_game_path(game_path)
                },
                Result::Ok(Option::None) => {
                    //Last element, cap it off
                    return builder;
                },
                Result::Err(_) => {
                    //Other end must've hung up, cap it off
                    return builder;
                },
            }
        }
    });

    (0..params.num_synthetic_training_games).into_par_iter()
    .map(|_| {
        let mut rng = rand::thread_rng();
        let game_path = params.generate_random_game_path(&mut rng);
        let annotated_game_path = game_path.annotate_path();

        annotated_game_path
    })
    .for_each_with(sender.clone(), |s, x| {
        s.send(Option::Some(x)).unwrap();
    });

    //Indicate that we've sent all the items we plan to
    sender.send(Option::None).unwrap();

    //Join the builder thread back to the main thread
    let maybe_builder = builder_thread_handle.join();
    match (maybe_builder) {
        Result::Ok(builder) => {
            let training_examples = builder.build();

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
        Result::Err(_) => {
            println!("Failed to join builder thread");
        }
    }

}

fn distill_game_data_command(params : Params, game_data_root : &str, training_data_output_path : &str) {
    //TODO: reimplement me!
    panic!();
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

    //Respect the freeze_non_injector_layers parameter
    for (var_name, tensor) in vs.variables() {
        if (params.freeze_non_injector_layers) {
            if (var_name.contains("injector")) {
                tensor.set_requires_grad(true);
            } else {
                tensor.set_requires_grad(false);
            }
        } else {
            tensor.set_requires_grad(true);
        }
    }

    let mut rng = rand::thread_rng();
    let data_path = Path::new(training_data_path);
    let maybe_training_examples = TrainingExamples::load(&data_path, tch::Device::Cpu);
    match (maybe_training_examples) {
        Result::Ok(training_examples) => { 
            println!("Successfully loaded training examples");
            let batch_split_training_examples = BatchSplitTrainingExamples::from_training_examples(
                training_examples,
                params.batch_size,
                params.held_out_validation_batches,
            );

            println!("Elaborating validation set");
            let validation_set = ValidationSet::from_batch_split_sketches(&params, &batch_split_training_examples);
            println!("Validation set elaborated. Starting training.");

            let adam = Adam {
                wd : params.weight_decay_factor,
                ..Adam::default()
            };
            let mut opt = adam.build(&vs, params.train_step_size).unwrap();
            let mut best_validation_loss = f64::INFINITY;
            loop {
                let (train_loss, validation_loss) = network_config.run_training_epoch(&params, &batch_split_training_examples, 
                                                                   &validation_set,
                                                                   &mut opt, device, &mut rng, &vs);
                let current_time = SystemTime::now();
                println!("train loss for the epoch: {} validation loss: {} current time: {:?}", 
                         train_loss, validation_loss, current_time);

                if (validation_loss < best_validation_loss) {
                    best_validation_loss = validation_loss;
                    println!("Validation loss is a new record! Writing out updated network configuration...");
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
            }
        },
        Result::Err(err) => {
            eprintln!("Failed to read training examples: {}", err);
            return;
        }
    }
}
