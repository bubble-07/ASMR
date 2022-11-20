use tch::{nn, nn::Init, nn::linear, nn::Module, Tensor, nn::Path, nn::Sequential, nn::LinearConfig};
use std::borrow::Borrow;
use crate::params::*;
use crate::neural_utils::*;
use crate::network_module::*;

///BiModule that takes a flattened matrix (dimension FLATTENED_MATRIX_DIM)
///for an input and a flattened matrix (dimension FLATTENED_MATRIX_DIM) for a target
///to descriptors of NUM_FEAT_MAPS size.
pub fn injector_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> BiConcatThenSequential {
    let vs = vs.borrow();
    let mut net = bi_concat_then_seq();
    let two_matrix_dim = 2 * params.get_flattened_matrix_dim();
    net = net.add(linear(
                     vs / "init_linear",
                     two_matrix_dim as i64,
                     params.num_feat_maps as i64,
                     Default::default()
                  ));
    net = net.add(residual_block(
                  vs / "residual_block",
                  params.num_injection_layers,
                  params.num_feat_maps));
    net
}

///Module that takes thre num_feat_maps-sized inputs,
///the first for the "left" element's track, the second for the "right" element's track,
///and the third for the global track, and yields a probability value of dimension 1 which
///will be a logit for that particular pairing
pub fn policy_extraction_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> TriConcatThenSequential {
    let vs = vs.borrow();
    let mut net = tri_concat_then_seq();
    let three_feat_dim = 3 * params.num_feat_maps;
    net = net.add(residual_block(
                  vs / "residual_block",
                  params.num_policy_extraction_layers,
                  three_feat_dim));

    //Softmax is translation-invariant, so we don't need bias here
    let output_config = LinearConfig {
        ws_init : Init::KaimingUniform,
        bs_init : Option::None,
        bias : false
    };

    net = net.add(linear(vs / "final_linear", three_feat_dim as i64, 1 as i64, 
                  output_config));
    net
}

///Network which takes injected matrix embeddings of num_feat_maps features
///by passing through a bunch of residual attention blocks after an initial linear
///mapping, where the residual attention blocks have an extra "global track"
///which is initialized with the average of the linear-mapped (input, target) pairs
///
///This is the network whose evaluation is only required once, at the root of the game tree.
///As a result, this network only ever processes the initial "root" set of matrices
pub fn root_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> ResidualAttentionStackWithGlobalTrack {
    residual_attention_stack_with_global_track(vs, params.num_main_net_layers, params.num_feat_maps)
}

///Network which takes an injected matrix embedding of num_feat_maps features
///and the peeling states for the entire root_net to a new num_feat_maps output for the "peel"
///
///This network is used during playouts to determine the next move after addition of a
///new matrix product.
pub fn peel_net<'a, T : Borrow<Path<'a>>>(params : &Params, vs : T) -> PeelStack { 
    peel_stack(vs, params.num_main_net_layers, params.num_feat_maps)
}

