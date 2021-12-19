use tch::{nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential};
use std::borrow::Borrow;
use crate::params::*;
use crate::neural_utils::*;

///BiModule that takes a flattened matrix (dimension FLATTENED_MATRIX_DIM)
///for an input and a flattened matrix (dimension FLATTENED_MATRIX_DIM) for a target
///to descriptors of NUM_FEAT_MAPS size.
pub fn injector_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let two_matrix_dim = 2 * params.get_flattened_matrix_dim();
    net = net.add(nn::linear(
                     vs / "init_linear",
                     two_matrix_dim,
                     params.num_feat_maps,
                     Default::default()
                  ));
    for i in 0..params.singleton_injection_layers {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      params.num_feat_maps));
    } 
    //Ensures that the scale of the output is the same as the output
    //scale of combiner net, roughly speaking
    net = net.add_fn(|xs| xs.tanh());
    net
}

///BiModule that takes two descriptors of NUM_FEAT_MAPS size
///and yields a combined descriptor of NUM_FEAT_MAPS size
pub fn combiner_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let feats = params.num_feat_maps * 2;
    for i in 0..params.combining_layers {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      feats));
    }
    net = net.add(nn::linear(
                      vs / format!("final_linear"),
                      feats,
                      params.num_feat_maps,
                      Default::default()
                 ));
    //Ensures that the scale of the output remains roughly the same
    //from combiner to combiner
    net = net.add_fn(|xs| xs.tanh());
    net
}

///BiModule that takes a descriptor of NUM_FEAT_MAPS size
///representing the combined state of a collection of matrices
///and a descriptor of NUM_FEAT_MAPS size representing the
///descriptor for a single matrix, and yields a descriptor
///of NUM_FEAT_MAPS * 2 size which will be dot-producted
///with the corresponding ("left"/"right") policy vector to yield
///a scalar policy value representing the "goodness" of the combination
pub fn half_policy_extraction_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let feats = params.num_feat_maps * 2;
    for i in 0..params.policy_extraction_layers {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      feats));
    }
    //We need this last linear layer because otherwise we could have
    //less expressive dot-products due to the Leaky RELU in the last
    //layer (albeit with a residual skip-connection, but relying
    //on that would mean greater reliance on the behavior
    //of earlier layers)
    net = net.add(nn::linear(
                      vs / format!("final_linear"),
                      feats,
                      feats,
                      Default::default()
                  ));
    net = net.add_fn(|xs| xs.tanh());
    net
}
