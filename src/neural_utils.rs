use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::Linear, nn::LinearConfig, nn::linear};
use std::borrow::Borrow;
use crate::params::*;

pub trait KModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor>;
}

#[derive(Debug)]
pub struct ResidualAttentionStackWithGlobalTrack {
    pub blocks : Vec<ResidualAttentionBlockWithGlobalTrack>
}

pub fn residual_attention_stack_with_global_track<'a, T : Borrow<Path<'a>>>(network_path : T,
    num_blocks : usize, num_layers_per_block : usize, full_dimension : usize)
    -> ResidualAttentionStackWithGlobalTrack {
    
    let network_path = network_path.borrow();

    let mut blocks = Vec::new();
    for i in 0..num_blocks {
        let block_path = network_path / format!("block{}", i);
        let block = residual_attention_block_with_global_track(block_path, num_layers_per_block, 
                                                        full_dimension);
        blocks.push(block);
    }
    ResidualAttentionStackWithGlobalTrack {
        blocks
    }
}

impl KModule for ResidualAttentionStackWithGlobalTrack {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let mut result = Vec::new();
        for x in xs.iter() {
            let x_clone = x.shallow_clone();
            result.push(x_clone)
        }
        for block in self.blocks.iter() {
            result = block.forward(&result);
        }
        result
    }
}

///A bilinear self-attention module on (K+1) inputs (the last of which is taken to be
///the "global" input), followed by a uniform mapping of a ResidualBlock on the first K
///inputs and a separate ResidualBlock on the "global" input. 
#[derive(Debug)]
pub struct ResidualAttentionBlockWithGlobalTrack {
    pub bilinear_self_attention : BilinearSelfAttention,
    pub mapped_residual_block : MappedModule<ResidualBlock>,
    pub global_residual_block : ResidualBlock
}


pub fn residual_attention_block_with_global_track<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      num_layers : usize, full_dimension : usize) ->
                                                                    ResidualAttentionBlockWithGlobalTrack {
    let network_path = network_path.borrow();
    let bilinear_self_attention = bilinear_self_attention(network_path / "bilinear_self_attention", full_dimension);
    let mapped_residual_block = residual_block(network_path / "mapped_residual_block", num_layers, full_dimension);
    let mapped_residual_block = map_module(mapped_residual_block);
    let global_residual_block = residual_block(network_path / "global_residual_block", num_layers, full_dimension);

    ResidualAttentionBlockWithGlobalTrack {
        bilinear_self_attention,
        mapped_residual_block,
        global_residual_block
    }
}

impl KModule for ResidualAttentionBlockWithGlobalTrack {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let after_attention = self.bilinear_self_attention.forward(xs);
        let mut residual_inputs = Vec::new();
        for i in 0..after_attention.len() {
            let x = &xs[i];
            let after_attention = &after_attention[i];
            let residual_input = x + after_attention;
            residual_inputs.push(residual_input);
        }
        let global_residual_input = residual_inputs.pop().unwrap();
        let global_residual_result = self.global_residual_block.forward(&global_residual_input);

        let mut result = self.mapped_residual_block.forward(&residual_inputs);
        result.push(global_residual_result);
        result
    }
}

pub trait BiModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor;
}

#[derive(Debug)]
pub struct Concat { }

#[derive(Debug)]
pub struct ConcatThenSequential { 
    bimod : Concat,
    seq : Sequential
}

impl BiModule for Concat {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor {
        let concatted = Tensor::concat(&[xs, ys], 1i64);
        concatted
    }
}

impl BiModule for ConcatThenSequential {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor {
        let init = self.bimod.forward(xs, ys);
        let rest = self.seq.forward(&init);
        rest
    }
}

impl ConcatThenSequential {
    pub fn add<M : Module + 'static>(self, layer : M) -> Self {
        let sequential = self.seq.add(layer);
        ConcatThenSequential {
            bimod : self.bimod,
            seq : sequential
        }
    }
    pub fn add_fn<F>(self, f : F) -> Self 
    where
        F : 'static + Fn(&Tensor) -> Tensor + Send,
    {
        self.add(tch::nn::func(f))
    }
}

pub fn concat_then_seq() -> ConcatThenSequential {
    let bimod = Concat { };
    let seq = nn::seq();
    ConcatThenSequential {
        bimod,
        seq
    }
}

#[derive(Debug)]
pub struct MappedModule<M : Module> {
    pub module : M
}

pub fn map_module<M : Module>(module : M) -> MappedModule<M> {
    MappedModule {
        module
    }
}

impl <M : Module> KModule for MappedModule<M> {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let mut results = Vec::new();
        for x in xs.iter() {
            let result = self.module.forward(x); 
            results.push(result);
        }
        results
    }
}

#[derive(Debug)]
pub struct BilinearSelfAttention {
    pub full_dimension : usize,
    ///=1/sqrt(full_dimension)
    pub scaling_factor : f32,
    ///Interaction matrix (Q^T K) of dimensions full_dimension x full_dimension
    pub interaction_matrix : Tensor,
    ///Linear layer from full_dimension -> full_dimension whose effect will be modulated by the
    ///interaction matrix
    pub value_extractor : nn::Linear
}

pub fn bilinear_self_attention<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      full_dimension : usize) -> BilinearSelfAttention {
    let network_path = network_path.borrow();
    let scaling_factor = 1.0f32 / (full_dimension as f32).sqrt();
    let dimensions = vec![full_dimension as i64, full_dimension as i64];

    let interaction_matrix = network_path.var("interaction_matrix", &dimensions, Init::KaimingUniform);

    //Our motivation for using this initialization is to zero out the residual branch output
    //in the larger network, to match the behavior of Normalizer-Free ResNets
    let value_extractor_config = LinearConfig {
        ws_init : Init::Const(0.0),
        bs_init : Option::Some(Init::Const(0.0)),
        bias : true
    };

    let value_extractor = nn::linear(network_path / "value_layer", full_dimension as i64,
                                     full_dimension as i64, value_extractor_config);

    BilinearSelfAttention {
        full_dimension,
        scaling_factor,
        interaction_matrix,
        value_extractor
    }
}

impl KModule for BilinearSelfAttention {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        //N x K x F
        let xs_forward = Tensor::stack(xs, 1);
        //N x F x K
        let xs_reverse = Tensor::stack(xs, 2);

        let scaled_interaction_matrix = self.scaling_factor * &self.interaction_matrix;

        //N x K x K
        let pre_softmax_weights = xs_forward.matmul(&scaled_interaction_matrix).matmul(&xs_reverse);
        let softmax_weights = pre_softmax_weights.softmax(2, Kind::Float);

        //N x K x F
        let values = self.value_extractor.forward(&xs_forward);
        
        //N x K x F
        let output_tensor = softmax_weights.matmul(&values);

        let outputs = output_tensor.unbind(1);
        outputs
    }
}

///A block of layers, all with residual skip-connections
#[derive(Debug)]
pub struct ResidualBlock {
    pub layers : Vec<LinearResidual>
}

pub fn residual_block<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      num_layers : usize, dim : usize) -> ResidualBlock {
    let network_path = network_path.borrow();

    let mut layers = Vec::new();
    for i in 0..num_layers {
        let layer_path = network_path / format!("layer{}", i);
        let layer = linear_residual(layer_path, dim);
        layers.push(layer);
    }
    ResidualBlock {
        layers
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs : &Tensor) -> Tensor {
        let mut result = xs.shallow_clone();
        for layer in self.layers.iter() {
            result = layer.forward(&result);
        }
        result
    }
}

///A single layer with a residual skip-connection
#[derive(Debug)]
pub struct LinearResidual {
    pub first_ws : Tensor,
    pub second_ws : Tensor,
    pub first_bs : Tensor,
    pub second_bs : Tensor
}

pub fn linear_residual<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      dim : usize) -> LinearResidual {
    let network_path = network_path.borrow();
    let bound = 1.0 / (dim as f64).sqrt();

    let first_bs = network_path.var("first_bias", &[dim as i64], Init::Uniform {lo : -bound, up : bound});
    let first_ws = network_path.var("first_weights", &[dim as i64, dim as i64], Init::KaimingUniform);

    //Inspired by normalizer-free ResNets, we initialize these to zero
    let second_bs = network_path.var("second_bias", &[dim as i64], Init::Const(0.0));
    let second_ws = network_path.var("second_weights", &[dim as i64, dim as i64], Init::Const(0.0));

    LinearResidual {
        first_ws,
        second_ws,
        first_bs,
        second_bs
    }
}

impl Module for LinearResidual {
    fn forward(&self, xs : &Tensor) -> Tensor {
        let pre_activation = xs.matmul(&self.first_ws.tr()) + &self.first_bs;
        let post_activation = pre_activation.leaky_relu();
        let post_weights = post_activation.matmul(&self.second_ws.tr()) + &self.second_bs;
        post_weights + xs
    }
}
