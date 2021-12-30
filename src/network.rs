use tch::{nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential, nn::Linear, nn::LinearConfig,
          nn::linear};
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
                     two_matrix_dim as i64,
                     params.num_feat_maps as i64,
                     Default::default()
                  ));
    for i in 0..params.num_injection_layers {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      params.num_feat_maps));
    } 
    net
}

///Module that takes two num_feat_maps-sized inputs, the first
///for a single element's track and the second for the global track,
///and yields a descriptor of num_feat_maps size which will be dot-producted with the corresponding ("left"/"right") policy
///vector to yield a scalar policy value representing the 'goodness" of the combination
pub fn half_policy_extraction_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ConcatThenSequential {
    let vs = vs.borrow();
    let mut net = concat_then_seq();
    let two_feat_dim = 2 * params.num_feat_maps;
    for i in 0..params.num_policy_extraction_layers {
        net = net.add(linear_residual(
                      vs / format!("layer_{}", i),
                      two_feat_dim));
    }
    net = net.add(linear(vs / "final_linear", two_feat_dim as i64, params.num_feat_maps as i64, 
                  LinearConfig::default()));
    //We need to bound the output here, or the gradients could potentially get too big.
    //We compensate for this by applying a post-scaling to the policy logits, so no worries.
    net = net.add_fn(|xs| xs.tanh());
    net
}

///KModule which takes num_feat_maps inputs provided by the injector_net above,
///and yields outputs of num_feat_maps features by passing through a bunch o' residual
///attention blocks
pub fn main_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ResidualAttentionStackWithGlobalTrack {
    residual_attention_stack_with_global_track(vs, params.num_blocks, params.num_layers_per_block,
                                        params.num_feat_maps)
}
