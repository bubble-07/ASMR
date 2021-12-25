use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::Linear, nn::LinearConfig, nn::linear};
use std::borrow::Borrow;
use crate::params::*;

pub trait KModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor>;
}

#[derive(Debug)]
pub struct ResidualMultiHeadAttentionStack {
    pub blocks : Vec<ResidualMultiHeadAttentionBlock>
}

pub fn residual_multi_head_attention_stack<'a, T : Borrow<Path<'a>>>(network_path : T,
    num_blocks : usize, num_layers_per_block : usize, full_dimension : usize, num_heads : usize)
    -> ResidualMultiHeadAttentionStack {
    
    let network_path = network_path.borrow();

    let mut blocks = Vec::new();
    for i in 0..num_blocks {
        let block_path = network_path / format!("block{}", i);
        let block = residual_multi_head_attention_block(block_path, num_layers_per_block, 
                                                        full_dimension, num_heads);
        blocks.push(block);
    }
    ResidualMultiHeadAttentionStack {
        blocks
    }
}

impl KModule for ResidualMultiHeadAttentionStack {
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

#[derive(Debug)]
pub struct ResidualMultiHeadAttentionBlock {
    pub multi_head_self_attention : MultiHeadSelfAttention,
    pub residual_block : MappedModule<ResidualBlock>
}

pub fn residual_multi_head_attention_block<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      num_layers : usize, full_dimension : usize, num_heads : usize) ->
                                                                    ResidualMultiHeadAttentionBlock {
    let network_path = network_path.borrow();
    let multi_head_self_attention = multi_head_self_attention(network_path / "multi_head_self_attention", full_dimension, num_heads);
    let residual_block = residual_block(network_path / "residual_block", num_layers, full_dimension);
    let mapped_residual_block = map_module(residual_block);

    ResidualMultiHeadAttentionBlock {
        multi_head_self_attention,
        residual_block : mapped_residual_block
    }
}

impl KModule for ResidualMultiHeadAttentionBlock {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let after_attention = self.multi_head_self_attention.forward(xs);
        let mut residual_inputs = Vec::new();
        for i in 0..after_attention.len() {
            let x = &xs[i];
            let after_attention = &after_attention[i];
            let residual_input = x + after_attention;
            residual_inputs.push(residual_input);
        }
        self.residual_block.forward(&residual_inputs)
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
pub struct MultiHeadSelfAttention {
    pub num_heads : usize,
    pub full_dimension : usize,
    ///head_dimension * num_heads = full_dimension
    pub head_dimension : usize,
    ///=1/sqrt(head_dimension)
    pub scaling_factor : f32,
    ///Linear layers from full_dimension -> head_dimension -- there are num_heads of them
    pub query_formers : Vec<Tensor>,
    pub key_formers : Vec<Tensor>,
    pub value_formers : Vec<Tensor>
}

pub fn multi_head_self_attention<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      full_dimension : usize, num_heads : usize) -> MultiHeadSelfAttention {
    let network_path = network_path.borrow();
    let head_dimension = full_dimension / num_heads;
    let scaling_factor = 1.0f32 / (head_dimension as f32).sqrt();

    let mut query_formers = Vec::new();
    let mut key_formers = Vec::new();
    let mut value_formers = Vec::new();

    for i in 0..num_heads {
        let query_name = format!("query_former{}", i);
        let key_name = format!("key_former{}", i);
        let value_name = format!("value_former{}", i);

        let dimensions = vec![head_dimension as i64, full_dimension as i64];
        let query_former = network_path.var(&query_name, &dimensions, Init::KaimingUniform);
        let key_former = network_path.var(&key_name, &dimensions, Init::KaimingUniform);

        //Inspired by normalization-free ResNets, we initialize these to zero
        let value_former = network_path.var(&value_name, &dimensions, Init::Const(0.0));

        query_formers.push(query_former);
        key_formers.push(key_former);
        value_formers.push(value_former);
    }
    MultiHeadSelfAttention {
        num_heads,
        full_dimension,
        head_dimension,
        scaling_factor,
        query_formers,
        key_formers,
        value_formers
    }
}

impl MultiHeadSelfAttention {
    ///Takes in K tensors of N examples each, full_dimension features [that is, K (N x full_dimension) tensors]
    ///and num_heads tensors for linear maps of full_dimension -> head_dimension,
    ///and returns num_heads tensors of dimensions N x K x head_dimension
    fn build_attention_input(&self, xs : &[Tensor], formers : &[Tensor]) -> Vec<Tensor> {
        let k = xs.len();
        
        let mut results = Vec::new();
        for i in 0..self.num_heads {
            let mut head_result_vec = Vec::new();
            for j in 0..k {
                let x = &xs[j];
                let former = &formers[i];
                let y = x.matmul(&former.tr());
                head_result_vec.push(y);
            }
            let head_result = Tensor::stack(&head_result_vec, 1); 
            results.push(head_result);
        }
        results
    }
}

impl KModule for MultiHeadSelfAttention {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let queries = self.build_attention_input(xs, &self.query_formers);
        let keys = self.build_attention_input(xs, &self.key_formers);
        let values = self.build_attention_input(xs, &self.value_formers);

        let mut head_results = Vec::new();
        for i in 0..self.num_heads {
            //Each of dimensions N x K x head_dimension
            let query = &queries[i];
            let key = &keys[i];
            let value = &values[i];

            let scaled_query = self.scaling_factor * query;
            let transposed_key = key.transpose(1, 2);

            //Dimensions are N x K x K
            let pre_softmax_weights = scaled_query.matmul(&transposed_key);
            let softmax_weights = pre_softmax_weights.softmax(2, Kind::Float);

            //Dimensions are N x K x head_dimension
            let head_result = softmax_weights.matmul(value);

            head_results.push(head_result);
        }

        //Dimensions N x K x full_dimension
        let result_tensor = Tensor::concat(&head_results, 2);
        let result = result_tensor.unbind(1);
        result
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
