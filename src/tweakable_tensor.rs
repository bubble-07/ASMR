use tch::{kind, Tensor};

///A Tensor which may or may not have been
///"tweaked" by adding some portion of another tensor
#[derive(Debug)]
pub enum TweakableTensor {
    Untweaked(Tensor),
    Tweaked(TweakedTensor),
}

impl From<Tensor> for TweakableTensor {
    fn from(tensor : Tensor) -> Self {
        TweakableTensor::Untweaked(tensor)
    }
}

impl TweakableTensor {
    pub fn bare_ref(&self) -> &Tensor {
        match (&self) {
            TweakableTensor::Untweaked(x) => x,
            _ => {
                panic!();
            },
        }
    }
    pub fn bare(self) -> Tensor {
        match (self) {
            TweakableTensor::Untweaked(x) => x,
            _ => {
                panic!();
            },
        }
    }
    pub fn tweaked(base_tensor : &Tensor, tweak_weight : &Tensor,
                   tweak : Tensor) -> TweakableTensor {
        TweakableTensor::Tweaked(TweakedTensor {
            base_tensor : base_tensor.shallow_clone(),
            tweak_weight : tweak_weight.shallow_clone(),
            tweak
        })
    }
    pub fn get(&self) -> Tensor {
        match (&self) {
            TweakableTensor::Untweaked(x) => x.shallow_clone(),
            TweakableTensor::Tweaked(x) => {
                x.get()
            }
        }
    }
}

#[derive(Debug)]
pub struct TweakedTensor {
    pub base_tensor : Tensor,
    pub tweak_weight : Tensor,
    pub tweak : Tensor,
}

impl TweakedTensor {
    pub fn get(&self) -> Tensor {
        &self.base_tensor + (&self.tweak_weight * &self.tweak)
    }
}

