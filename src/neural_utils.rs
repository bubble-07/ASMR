use tch::{nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential};
use std::borrow::Borrow;
use crate::params::*;

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
pub struct LinearResidual {
    pub first_ws : Tensor,
    pub second_ws : Tensor,
    pub first_bs : Tensor,
    pub second_bs : Tensor
}

pub fn linear_residual<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      dim : i64) -> LinearResidual {
    let network_path = network_path.borrow();
    let bound = 1.0 / (dim as f64).sqrt();

    let first_bs = network_path.var("first_bias", &[dim], Init::Uniform {lo : -bound, up : bound});
    let second_bs = network_path.var("second_bias", &[dim], Init::Uniform {lo : -bound, up : bound});
    let first_ws = network_path.var("first_weights", &[dim, dim], Init::KaimingUniform);
    let second_ws = network_path.var("second_weights", &[dim, dim], Init::KaimingUniform);
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
