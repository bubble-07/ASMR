use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::Linear, nn::LinearConfig, nn::linear};
use std::borrow::Borrow;
use crate::params::*;

pub trait KModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor>;
}

#[derive(Debug)]
pub struct ResidualAttentionStackWithGlobalTrack {
    pub layers : Vec<ResidualAttentionLayerWithGlobalTrack>
}

pub fn residual_attention_stack_with_global_track<'a, T : Borrow<Path<'a>>>(network_path : T,
    num_layers : usize, full_dimension : usize)
    -> ResidualAttentionStackWithGlobalTrack {
    
    let network_path = network_path.borrow();

    let mut layers = Vec::new();
    for i in 0..num_layers {
        let layer_path = network_path / format!("layer{}", i);
        let layer = residual_attention_layer_with_global_track(layer_path, full_dimension);
        layers.push(layer);
    }
    ResidualAttentionStackWithGlobalTrack {
        layers
    }
}

impl KModule for ResidualAttentionStackWithGlobalTrack {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let mut result = Vec::new();
        for x in xs.iter() {
            let x_clone = x.shallow_clone();
            result.push(x_clone)
        }
        for layer in self.layers.iter() {
            result = layer.forward(&result);
        }
        result
    }
}

///A bilinear self-attention module on (K+1) inputs (the last of which is taken to be
///the "global" input), followed by a uniform mapping of a ResidualBlock on the first K
///inputs and a separate ResidualBlock on the "global" input. 
///
///A residually-mapped function which consists of the sum of bilinear_self_attention + linear_transform +
///bias mapped into a a leaky relu nonlinearity, followed by an arbitrary linear transformation of
///the output (initialized to zero out the output), added back onto the residual branch.
///The attention components are taken to be uniform across all (K+1) inputs (the last of which
///is taken to be the "global" input), but linear and bias component parameters are split between
///the uniform K elements and the extra "global" element.
#[derive(Debug)]
pub struct ResidualAttentionLayerWithGlobalTrack {
    pub bilinear_self_attention : BilinearSelfAttention,
    pub pre_linear_general : nn::Linear,
    pub pre_linear_global : nn::Linear,
    pub post_linear_general : nn::Linear,
    pub post_linear_global : nn::Linear
}


impl KModule for ResidualAttentionLayerWithGlobalTrack {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        let mut general_inputs = Vec::new();
        for i in 0..(xs.len() - 1) {
            general_inputs.push(xs[i].shallow_clone());
        }
        let global_input = xs[xs.len() - 1].shallow_clone();

        let mut after_attention_general = self.bilinear_self_attention.forward(xs);
        let after_attention_global = after_attention_general.pop().unwrap();

        let mut result = Vec::new();
        for (general_input, after_attention_general) in general_inputs.iter()
                                                                      .zip(after_attention_general.iter()) {
            let after_linear = self.pre_linear_general.forward(general_input);
            let sum = &after_linear + after_attention_general;
            let post_activation = sum.leaky_relu();
            let output_general = self.post_linear_general.forward(&post_activation) + general_input;
            result.push(output_general);
        }

        let sum_global = &after_attention_global + &self.pre_linear_global.forward(&global_input);
        let post_activation_global = sum_global.leaky_relu();
        let output_global = self.post_linear_global.forward(&post_activation_global) + &global_input;
        result.push(output_global);

        result
    }
}

pub fn residual_attention_layer_with_global_track<'a, T : Borrow<Path<'a>>>(network_path : T, 
                    full_dimension : usize) -> ResidualAttentionLayerWithGlobalTrack {
    let network_path = network_path.borrow();
    let bilinear_self_attention = bilinear_self_attention(network_path / "bilinear_self_attention", full_dimension);

    let pre_linear_general = nn::linear(network_path / "pre_linear_general", 
                                        full_dimension as i64, full_dimension as i64, Default::default());
    let pre_linear_global = nn::linear(network_path / "pre_linear_global",
                                        full_dimension as i64, full_dimension as i64, Default::default());

    //These are initialized to zero in the style of normalization-free ResNets
    let post_linear_config = LinearConfig {
        ws_init : Init::Const(0.0),
        bs_init : Option::Some(Init::Const(0.0)),
        bias : true
    };

    let post_linear_general = nn::linear(network_path / "post_linear_general",
                                         full_dimension as i64, full_dimension as i64, post_linear_config);
    let post_linear_global = nn::linear(network_path / "post_linear_global",
                                         full_dimension as i64, full_dimension as i64, post_linear_config);

    ResidualAttentionLayerWithGlobalTrack {
        bilinear_self_attention,
        pre_linear_general,
        pre_linear_global,
        post_linear_general,
        post_linear_global
    }
}

pub trait TriModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &Tensor, ys : &Tensor, zs : &Tensor) -> Tensor;
}

#[derive(Debug)]
pub struct TriConcat { }

#[derive(Debug)]
pub struct TriConcatThenSequential {
    trimod : TriConcat,
    seq : Sequential
}

impl TriModule for TriConcat {
    fn forward(&self, xs : &Tensor, ys : &Tensor, zs : &Tensor) -> Tensor {
        let concatted = Tensor::concat(&[xs, ys, zs], 1i64);
        concatted
    }
}

impl TriModule for TriConcatThenSequential {
    fn forward(&self, xs : &Tensor, ys : &Tensor, zs : &Tensor) -> Tensor {
        let init = self.trimod.forward(xs, ys, zs);
        let rest = self.seq.forward(&init);
        rest
    }
}

impl TriConcatThenSequential {
    pub fn add<M : Module + 'static>(self, layer : M) -> Self {
        let sequential = self.seq.add(layer);
        TriConcatThenSequential {
            trimod : self.trimod,
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

pub trait BiModule : std::fmt::Debug + Send {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor;
}

#[derive(Debug)]
pub struct BiConcat { }

#[derive(Debug)]
pub struct BiConcatThenSequential { 
    bimod : BiConcat,
    seq : Sequential
}

impl BiModule for BiConcat {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor {
        let concatted = Tensor::concat(&[xs, ys], 1i64);
        concatted
    }
}

impl BiModule for BiConcatThenSequential {
    fn forward(&self, xs : &Tensor, ys : &Tensor) -> Tensor {
        let init = self.bimod.forward(xs, ys);
        let rest = self.seq.forward(&init);
        rest
    }
}

impl BiConcatThenSequential {
    pub fn add<M : Module + 'static>(self, layer : M) -> Self {
        let sequential = self.seq.add(layer);
        BiConcatThenSequential {
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

pub fn bi_concat_then_seq() -> BiConcatThenSequential {
    let bimod = BiConcat { };
    let seq = nn::seq();
    BiConcatThenSequential {
        bimod,
        seq
    }
}

pub fn tri_concat_then_seq() -> TriConcatThenSequential {
    let trimod = TriConcat { };
    let seq = nn::seq();
    TriConcatThenSequential {
        trimod,
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
    ///Linear transform from full_dimension -> full_dimension whose effect will be modulated by the
    ///interaction matrix
    pub value_extractor : Tensor,
    ///Left-bias for the interaction matrix
    pub left_bias : Tensor,
    ///Right-bias for the interaction matrix
    pub right_bias : Tensor
}

pub fn bilinear_self_attention<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      full_dimension : usize) -> BilinearSelfAttention {
    let network_path = network_path.borrow();
    let scaling_factor = 1.0f32 / (full_dimension as f32).sqrt();
    let dimensions = vec![full_dimension as i64, full_dimension as i64];

    let interaction_matrix = network_path.var("interaction_matrix", &dimensions, Init::KaimingUniform);
    let value_extractor = network_path.var("value_extractor", &dimensions, Init::KaimingUniform);

    let bias_dimensions = vec![full_dimension as i64];

    let bias_init = Init::Uniform {
        lo : -scaling_factor as f64,
        up : scaling_factor as f64
    };
    let left_bias = network_path.var("left_bias", &bias_dimensions, bias_init);
    let right_bias = network_path.var("right_bias", &bias_dimensions, bias_init);

    BilinearSelfAttention {
        full_dimension,
        scaling_factor,
        interaction_matrix,
        value_extractor,
        left_bias,
        right_bias
    }
}

impl KModule for BilinearSelfAttention {
    fn forward(&self, xs : &[Tensor]) -> Vec<Tensor> {
        //N x K x F
        let xs_forward = Tensor::stack(xs, 1);
        let xs_forward_biased = &xs_forward + &self.left_bias;

        //N x F x K
        let xs_reverse_biased = (&xs_forward + &self.right_bias).transpose(1, 2);

        let scaled_interaction_matrix = self.scaling_factor * &self.interaction_matrix;

        //N x K x K
        let pre_softmax_weights = xs_forward_biased.matmul(&scaled_interaction_matrix).matmul(&xs_reverse_biased);
        let softmax_weights = pre_softmax_weights.softmax(2, Kind::Float);

        //N x K x F
        let values = xs_forward.matmul(&self.value_extractor);
        
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
