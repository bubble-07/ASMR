use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::Linear, nn::LinearConfig, nn::linear};
use std::borrow::Borrow;

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
