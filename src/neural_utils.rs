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
        let concatted = Tensor::concat(&[xs, ys], 0i64);
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
    pub ws : Tensor,
    pub bs : Tensor
}

pub fn linear_residual<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      dim : i64) -> LinearResidual {
    let network_path = network_path.borrow();
    let bound = 1.0 / (dim as f64).sqrt();
    let bs_init = Init::Uniform {
        lo: -bound,
        up: bound
    };
    let bs = network_path.var("bias", &[dim], bs_init);
    let ws = network_path.var("weight", &[dim, dim], Init::KaimingUniform);
    LinearResidual {
        ws,
        bs
    }
}

impl Module for LinearResidual {
    fn forward(&self, xs : &Tensor) -> Tensor {
        let pre_activation = xs.matmul(&self.ws.tr()) + &self.bs;
        let post_activation = pre_activation.leaky_relu();
        post_activation + xs
    }
}
