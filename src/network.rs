use tch::{nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential};
use std::borrow::Borrow;
use crate::params::*;
use crate::neural_utils::*;

///BiModule that takes a flattened matrix (dimension FLATTENED_MATRIX_DIM)
///for an input and a flattened matrix (dimension FLATTENED_MATRIX_DIM) for a target
///to descriptors of NUM_FEAT_MAPS size.
pub fn injector_net<'a, T : Borrow<Path<'a>>>(vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let two_matrix_dim = 2 * FLATTENED_MATRIX_DIM;
    net = net.add(nn::linear(
                     vs / "init_linear",
                     two_matrix_dim,
                     NUM_FEAT_MAPS,
                     Default::default()
                  ));
    for i in 0..SINGLETON_INJECTION_LAYERS {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      NUM_FEAT_MAPS));
    } 
    net
}

///BiModule that takes two descriptors of NUM_FEAT_MAPS size
///and yields a combined descriptor of NUM_FEAT_MAPS size
pub fn combiner_net<'a, T : Borrow<Path<'a>>>(vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let feats = NUM_FEAT_MAPS * 2;
    for i in 0..COMBINING_LAYERS {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      feats));
    }
    net = net.add(nn::linear(
                      vs / format!("final_linear"),
                      feats,
                      NUM_FEAT_MAPS,
                      Default::default()
                 ));
    net
}

///BiModule that takes a descriptor of NUM_FEAT_MAPS size
///representing the combined state of a collection of matrices
///and a descriptor of NUM_FEAT_MAPS size representing the
///descriptor for a single matrix, and yields a descriptor
///of NUM_FEAT_MAPS * 2 size which will be dot-producted
///with the corresponding ("left"/"right") policy vector to yield
///a scalar policy value representing the "goodness" of the combination
pub fn half_policy_extraction_net<'a, T : Borrow<Path<'a>>>(vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let feats = NUM_FEAT_MAPS * 2;
    for i in 0..POLICY_EXTRACTION_LAYERS {
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
    net
}
