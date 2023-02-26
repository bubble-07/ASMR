use tch::{no_grad_guard, nn, nn::Init, nn::Module, Tensor, nn::Sequential, kind::Kind,
          kind::Element, nn::Optimizer, IndexOp, Device};

use crate::training_examples::*;
use crate::batch_split_training_examples::*;
use crate::playout_sketches::*;
use crate::params::*;

pub struct ValidationSet {
    weights_and_bundles : Vec<Vec<(f64, PlayoutBundle)>>,
}

impl ValidationSet {
    pub fn from_batch_split_sketches(params : &Params, 
                                 batch_split_training_examples : &BatchSplitTrainingExamples<PlayoutSketchBundle>) -> Self {
        let device_for_computations = params.get_device();
        let weights_and_bundles = batch_split_training_examples.iter_validation_batches(device_for_computations)
                                                               .map(|x| Self::process_weights_and_bundles(params, x))
                                                               .collect();
        
        Self {
            weights_and_bundles,
        }
    }
    fn process_weights_and_bundles(params : &Params, weights_and_bundles : Vec<(f64, PlayoutSketchBundle)>) -> Vec<(f64, PlayoutBundle)> {
        let mut result = Vec::new();
        for (weight, bundle) in weights_and_bundles {
            let elaborated_bundle = PlayoutBundle::from_sketch_bundle(params, bundle);
            let standardized_bundle = elaborated_bundle.standardize();
            let cpu_bundle = standardized_bundle.to_device(Device::Cpu);
            result.push((weight, cpu_bundle));
        }
        result
    }
    fn change_device(weights_and_bundles : &[(f64, PlayoutBundle)], device : Device) -> Vec<(f64, PlayoutBundle)> {
        let mut result = Vec::new();
        for (weight, bundle) in weights_and_bundles {
            result.push((*weight, bundle.to_device(device)));
        }
        result
    }
    pub fn iter_validation_batches<'a>(&'a self, device : Device) ->
                                  impl Iterator<Item = Vec<(f64, PlayoutBundle)>> + 'a {
        self.weights_and_bundles.iter()
                                .map(move |weights_and_bundles| Self::change_device(weights_and_bundles, device))
    }
}
