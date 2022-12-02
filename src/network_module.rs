use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::LinearConfig};
use std::borrow::Borrow;
use crate::tweakable_tensor::*;

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

///A layer which takes _matrices_ as input and yields
///a bilinear-feature-mapped output of the form
///A M B for trainable matries A and B. The input
///and the output are both flattened matrices,
///and both the input and output size must be
///perfect squares.
pub struct BilinearMatrixSketch {
    //sqrt(out_dim) x sqrt(in_dim)
    pub left_matrix : Tensor,
    //sqrt(in_dim) x sqrt(out_dim)
    pub right_matrix : Tensor,
    pub sqrt_in_dim : usize,
    pub sqrt_out_dim : usize,
    pub in_dim : usize,
    pub out_dim : usize,
}

pub fn bilinear_matrix_sketch<'a, T : Borrow<Path<'a>>>(network_path : T,
                             in_dim : usize, out_dim : usize) -> BilinearMatrixSketch {
    panic!();
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

pub fn simple_linear_tweak<'a, T : Borrow<Path<'a>>>(network_path : T,
                                                     base_simple_linear : &SimpleLinear)
                                                    -> SimpleLinear {
    let network_path = network_path.borrow();
    let tweak_path = network_path / "tweak";

    let in_out_dim = base_simple_linear.in_out_dim;
    let config = base_simple_linear.config.clone();

    //Unlike the base part, we initialize the tweak to always
    //have non-zero components, because if they were all zero,
    //we'd run into an immediate issue with the tweak weight
    //and the weights here would be zero, which would lead to
    //an inescapable saddle-point
    let tweak_simple_linear = simple_linear(tweak_path, in_out_dim, Default::default());

    let ws = TweakableTensor::tweaked(base_simple_linear.ws.bare_ref(),
                                      0.0 * tweak_simple_linear.ws.bare());
    let bs = if (base_simple_linear.bs.is_some()) {
                Option::Some(TweakableTensor::tweaked(base_simple_linear.bs.as_ref().unwrap().bare_ref(),
                             0.0 * tweak_simple_linear.bs.unwrap().bare()))
             } else {
                 Option::None
             };
    SimpleLinear {
        in_out_dim,
        config,
        ws,
        bs
    }
}

///Our own implementation of a linear layer,
///to remove the stupid transpose operation
///in the case where the in and out dimensions are
///the same.
#[derive(Debug)]
pub struct SimpleLinear {
    in_out_dim : i64,
    config : LinearConfig,
    pub ws : TweakableTensor,
    pub bs : Option<TweakableTensor>,
}

pub fn simple_linear<'a, T : Borrow<Path<'a>>>(network_path : T,
                                        in_out_dim : i64,
                                        c : LinearConfig) -> SimpleLinear {
    let config = c.clone();
    let neural_net_linear = nn::linear(network_path, in_out_dim, in_out_dim, c);
    SimpleLinear {
        in_out_dim,
        config,
        ws : TweakableTensor::from(neural_net_linear.ws),
        bs : neural_net_linear.bs.map(|x| TweakableTensor::from(x)),
    }
}

impl Module for SimpleLinear {
    fn forward(&self, xs : &Tensor) -> Tensor {
        let ws = self.ws.get();
        let bs = self.bs.as_ref().map(|x| x.get());
        xs.linear(&ws, bs.as_ref())
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

    //Inspired by ReZero [batchnorm-free resnets], we initialize these to zero
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
        let pre_activation = xs.linear(&self.first_ws, Option::Some(&self.first_bs));
        let post_activation = pre_activation.leaky_relu();
        let post_weights = post_activation.linear(&self.second_ws, Option::Some(&self.second_bs));
        post_weights + xs
    }
}
