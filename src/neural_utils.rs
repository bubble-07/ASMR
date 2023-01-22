use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::LinearConfig};
use std::borrow::Borrow;
use crate::network_module::{SimpleLinear, simple_linear, simple_linear_tweak};
use crate::params::*;
use crate::peeling_states::*;
use crate::tweakable_tensor::*;

#[derive(Debug)]
pub struct PeelStack {
    pub layers : Vec<PeelLayer>
}

impl PeelStack {
    //Given a stack of per-layer [pre-] activations, yields the peeling states
    //layer-by-layer
    pub fn forward_to_peel_state(&self, layer_activations : &[Tensor]) -> PeelLayerStates {
        let mut result = Vec::new();
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let pre_activation = &layer_activations[i];
            let peeling_state = layer.forward_to_peel_state(pre_activation);
            result.push(peeling_state);
        }
        PeelLayerStates::new(result)
    }
    //Yields peeling tracks and the new output activation
    pub fn peel_forward_diff(&self, peel_layer_states : &PeelLayerStates, x : &Tensor) ->
        (PeelTrackStates, Tensor) {
        let mut activation = x.shallow_clone();
        let mut peeling_tracks = Vec::new();
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let peeling_state = peel_layer_states.get_layer_state(i);
            let (peeling_track, post_activation) = layer.peel_forward(&peeling_state, &activation);
            activation = post_activation;

            peeling_tracks.push(peeling_track);
        }
        let peeling_tracks = PeelTrackStates::new(peeling_tracks);
        (peeling_tracks, activation)
    }
    //Given current per-layer peel layer states, and the pre-activation for a new
    //"peel" track, yields the updated peel layer states incorporating the new track
    //and the final activation map out of the peel
    pub fn peel_forward(&self, peel_layer_states : PeelLayerStates, x : &Tensor) -> (PeelLayerStates, Tensor) {
        let (peeling_tracks, activation) = self.peel_forward_diff(&peel_layer_states, x);

        let updated_peeling_states = peel_layer_states.push_tracks(&peeling_tracks);
        (updated_peeling_states, activation)
    }
}

pub fn peel_stack<'a, T : Borrow<Path<'a>>>(network_path : T, 
                                            main_net_stack : &ResidualAttentionStackWithGlobalTrack)
                                            -> PeelStack {
    let network_path = network_path.borrow();

    let mut layers = Vec::new();
    for (i, main_net_layer) in main_net_stack.layers.iter().enumerate() {
        let layer_path = network_path / format!("layer{}", i);
        let layer = peel_layer(layer_path, main_net_layer);
        layers.push(layer);
    }
    PeelStack {
        layers,
    }
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

impl ResidualAttentionStackWithGlobalTrack {
    //N x K x F for input xs
    //Output is input activations indexed by layer, with an
    //additional last tensor for the output activation
    pub fn forward(&self, xs : &Tensor) -> Vec<Tensor> {
        let mut result = Vec::new();
        let mut activation = xs.shallow_clone();
        result.push(activation.shallow_clone());

        for layer in self.layers.iter() {
            activation = layer.forward(&activation);
            result.push(activation.shallow_clone());
        }

        result
    }
}

#[derive(Debug)]
pub struct PeelLayer {
    pub read_only_attention : BilinearSelfAttention,
    pub pre_linear : SimpleLinear,
    pub post_linear : SimpleLinear
}

impl PeelLayer {
    pub fn forward_to_peel_state(&self, xs_forward : &Tensor) -> PeelLayerState {
        let layer_normed = layer_norm(xs_forward);
        self.read_only_attention.forward_to_peel_state(&layer_normed)
    }

    pub fn peel_forward(&self, peel_layer_state : &PeelLayerState, x : &Tensor) -> (PeelTrackState, Tensor) {
        let layer_normed = layer_norm(x);
        let (peel_track_state, after_attention) = 
            self.read_only_attention.peel_forward(peel_layer_state, &layer_normed);
        let after_linear = self.pre_linear.forward(&x);
        let sum = &after_linear + &after_attention;
        let post_activation = sum.leaky_relu();
        let output = self.post_linear.forward(&post_activation) + x;
        (peel_track_state, output)
    }
}

//A peel layer _is_ a tweak of a ResidualAttentionLayerWithGlobalTrack,
//tweaking the "general matrix" functions
pub fn peel_layer<'a, T : Borrow<Path<'a>>>(network_path : T, 
                                            main_net_layer : &ResidualAttentionLayerWithGlobalTrack)
                          -> PeelLayer {
    let network_path = network_path.borrow();
    let read_only_attention = bilinear_self_attention_tweak(network_path / "read_only_attention",
                                                            &main_net_layer.bilinear_self_attention);
    let pre_linear = simple_linear_tweak(network_path / "pre_linear",
                                         &main_net_layer.pre_linear_general);
    let post_linear = simple_linear_tweak(network_path / "post_linear",
                                          &main_net_layer.post_linear_general);
    PeelLayer {
        read_only_attention,
        pre_linear,
        post_linear
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
    pub pre_linear_general : SimpleLinear,
    pub pre_linear_global : SimpleLinear,
    pub post_linear_general : SimpleLinear,
    pub post_linear_global : SimpleLinear
}

//Layer norms a tensor's last dimension [which is assumed to
//be the feature-sized dimension
fn layer_norm(tensor : &Tensor) -> Tensor {
    let last_dim = tensor.size()[tensor.size().len() - 1];
    tensor.layer_norm(&[last_dim], Option::<&Tensor>::None, Option::<&Tensor>::None, 0.000001, true)
}


impl ResidualAttentionLayerWithGlobalTrack {
    pub fn forward(&self, xs_forward : &Tensor) -> Tensor {
        let peel_layer_state = self.forward_to_peel_state(xs_forward);
        self.forward_from_peel_state(xs_forward, &peel_layer_state)
    }

    //xs : N x K x F
    pub fn forward_to_peel_state(&self, xs_forward : &Tensor) -> PeelLayerState {
        let layer_normed = layer_norm(xs_forward);
        self.bilinear_self_attention.forward_to_peel_state(&layer_normed)
    }

    //xs : N x K x F
    pub fn forward_from_peel_state(&self, xs_forward : &Tensor, peel_layer_state : &PeelLayerState) -> Tensor {
        let s = xs_forward.size();
        let (n, k, f) = (s[0], s[1], s[2]);
        let k_minus_one = Option::Some(k - 1);

        let layer_normed = layer_norm(xs_forward);

        let general_inputs = xs_forward.slice(1, Option::None, k_minus_one, 1);
        let global_input = xs_forward.slice(1, k_minus_one, Option::None, 1);
        let global_input = global_input.reshape(&[n, f]);

        //Compute attention
        let after_attention = self.bilinear_self_attention.forward_from_peel_state(&layer_normed, peel_layer_state);
        let after_attention_general = after_attention.slice(1, Option::None, k_minus_one, 1);
        let after_attention_global = after_attention.slice(1, k_minus_one, Option::None, 1);
        let after_attention_global = after_attention_global.reshape(&[n, f]);
        
        //Derive the general outputs
        //(N * (K - 1)) x F
        let leading_dim = n * (k - 1);
        let general_inputs = general_inputs.reshape(&[leading_dim, f]);
        let after_attention_general = after_attention_general.reshape(&[leading_dim, f]);
        let after_linear = self.pre_linear_general.forward(&general_inputs);
        let sum = &after_linear + after_attention_general;
        let post_activation = sum.leaky_relu();
        let general_outputs = self.post_linear_general.forward(&post_activation) + general_inputs;

        let general_outputs = general_outputs.reshape(&[n, k - 1, f]);

        //Derive the global output
        let sum_global = &after_attention_global + &self.pre_linear_global.forward(&global_input);
        let post_activation_global = sum_global.leaky_relu();
        let output_global = self.post_linear_global.forward(&post_activation_global) + &global_input;
        let output_global = output_global.reshape(&[n, 1, f]);

        let result_tensor = Tensor::concat(&[general_outputs, output_global], 1);
        result_tensor
    }
}

pub fn residual_attention_layer_with_global_track<'a, T : Borrow<Path<'a>>>(network_path : T, 
                    full_dimension : usize) -> ResidualAttentionLayerWithGlobalTrack {
    let network_path = network_path.borrow();
    let bilinear_self_attention = bilinear_self_attention(network_path / "bilinear_self_attention", full_dimension);

    let pre_linear_general = simple_linear(network_path / "pre_linear_general", 
                                        full_dimension as i64, Default::default());
    let pre_linear_global = simple_linear(network_path / "pre_linear_global",
                                        full_dimension as i64, Default::default());

    //These are initialized to zero in the style of ReZero (normalization-free ResNets)
    let post_linear_config = LinearConfig {
        ws_init : Init::Const(0.0),
        bs_init : Option::Some(Init::Const(0.0)),
        bias : true
    };

    let post_linear_general = simple_linear(network_path / "post_linear_general",
                                         full_dimension as i64, post_linear_config);
    let post_linear_global = simple_linear(network_path / "post_linear_global",
                                         full_dimension as i64, post_linear_config);

    ResidualAttentionLayerWithGlobalTrack {
        bilinear_self_attention,
        pre_linear_general,
        pre_linear_global,
        post_linear_general,
        post_linear_global
    }
}

#[derive(Debug)]
pub struct BilinearSelfAttention {
    pub full_dimension : usize,
    ///=1/sqrt(full_dimension)
    pub scaling_factor : f64,
    ///Interaction matrix (Q^T K) of dimensions full_dimension x full_dimension
    pub interaction_matrix : TweakableTensor,
    ///Linear transform from full_dimension -> full_dimension whose effect will be modulated by the
    ///interaction matrix
    pub value_extractor : TweakableTensor,
    ///Left-bias for the interaction matrix
    pub left_bias : TweakableTensor,
    ///Right-bias for the interaction matrix
    pub right_bias : TweakableTensor
}

///A Bilinear self attention module whose parameters are a small tweak
///from the parameters of another
pub fn bilinear_self_attention_tweak<'a, T : Borrow<Path<'a>>>(network_path : T,
                      base_attention : &BilinearSelfAttention) -> BilinearSelfAttention {
    let network_path = network_path.borrow();
    let tweak_path = network_path / "tweak";

    let full_dimension = base_attention.full_dimension;
    let scaling_factor = base_attention.scaling_factor;

    let tweak_attention = bilinear_self_attention(tweak_path, full_dimension);

    //Initialize tweak weight to zero in the interest of possibly keeping the
    //variability in parameters around for later [if useful]
    let interaction_matrix = TweakableTensor::tweaked(base_attention.interaction_matrix.bare_ref(),
                                                      0.0 * tweak_attention.interaction_matrix.bare());
    let value_extractor = TweakableTensor::tweaked(base_attention.value_extractor.bare_ref(),
                                                   0.0 * tweak_attention.value_extractor.bare());
    let left_bias = TweakableTensor::tweaked(base_attention.left_bias.bare_ref(),
                                             0.0 * tweak_attention.left_bias.bare());

    let right_bias = TweakableTensor::tweaked(base_attention.right_bias.bare_ref(),
                                             0.0 * tweak_attention.right_bias.bare());

    BilinearSelfAttention {
        full_dimension,
        scaling_factor,
        interaction_matrix,
        value_extractor,
        left_bias,
        right_bias
    }
}

pub fn bilinear_self_attention<'a, T : Borrow<Path<'a>>>(network_path : T, 
                      full_dimension : usize) -> BilinearSelfAttention {
    let network_path = network_path.borrow();
    let scaling_factor = 1.0f64 / (full_dimension as f64).sqrt();
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

    let interaction_matrix = TweakableTensor::from(interaction_matrix);
    let value_extractor = TweakableTensor::from(value_extractor);
    let left_bias = TweakableTensor::from(left_bias);
    let right_bias = TweakableTensor::from(right_bias);

    BilinearSelfAttention {
        full_dimension,
        scaling_factor,
        interaction_matrix,
        value_extractor,
        left_bias,
        right_bias
    }
}

impl BilinearSelfAttention {
    ///Given the states of all the other peel layers and an x for activation,
    ///yields the state of x's track after running it through the attentional layer
    pub fn peel_forward(&self, peel_layer_state : &PeelLayerState, x : &Tensor) -> (PeelTrackState, Tensor) {
        let left_bias = self.left_bias.get();
        let right_bias = self.right_bias.get();
        let value_extractor = self.value_extractor.get();

        //TODO: This is expensive, for some reason [due to matrix-multiplies]
        //make those faster!
        
        //N x F
        let x_left_biased = x + &left_bias;
        //N x 1 x F
        let x_left_biased = x_left_biased.unsqueeze(1);

        //Only really need the forward-biased side to determine softmax
        //weights for reading, since in "forward" below, for example,
        //the soft-maxing [and hence, the "reading"] happens on the reverse-biased elements
        
        //interactions is N x F x K
        
        //Determine forward interaction values
        //N x 1 x K
        let softmax_weights_unnormalized = x_left_biased.matmul(&peel_layer_state.interactions);
        //N x 1 x K
        let softmax_weights = softmax_weights_unnormalized.softmax(2, Kind::Float);

        //N x 1 x F
        let activation = softmax_weights.matmul(&peel_layer_state.values);
        //N x F
        let activation = activation.squeeze_dim(1);


        //Compute value for lookup from later peels -- N x F
        let value = x.matmul(&value_extractor);

        //The right-bias does enter in to updating the new interaction transforms
        //for the returned peel track state, tho
        //N x F
        let x_right_biased = x + &right_bias;
        //N x F x 1
        let x_right_biased = x_right_biased.unsqueeze(2);
        let scaled_interaction_matrix = &peel_layer_state.scaled_interaction_matrix;

        //N x F
        let interaction = scaled_interaction_matrix.matmul(&x_right_biased).squeeze_dim(2);

        let peel_track_state = PeelTrackState {
            value,
            interaction
        };

        (peel_track_state, activation)
    }

    //xs : N x K x F
    //Computes the PeelLayerState from the input tensor
    pub fn forward_to_peel_state(&self, xs : &Tensor) -> PeelLayerState {
        let right_bias = self.right_bias.get();
        let interaction_matrix = self.interaction_matrix.get();
        let value_extractor = self.value_extractor.get();

        //N x F x K
        let xs_right_biased = (xs + &right_bias).transpose(1, 2);

        let scaled_interaction_matrix = self.scaling_factor * &interaction_matrix;

        //N x K x F
        let values = xs.matmul(&value_extractor);

        //N x F x K
        let interactions = scaled_interaction_matrix.matmul(&xs_right_biased);

        PeelLayerState {
            values,
            interactions,
            scaled_interaction_matrix,
        }
    }
    //Computes the activation from the input tensor and the peel layer state
    pub fn forward_from_peel_state(&self, xs_forward : &Tensor, peel_layer_state : &PeelLayerState) -> Tensor {
        let left_bias = self.left_bias.get();

        //N x K x F
        let xs_left_biased = xs_forward + &left_bias;

        //N x K x K
        let pre_softmax_weights = xs_left_biased.matmul(&peel_layer_state.interactions);
        let softmax_weights = pre_softmax_weights.softmax(2, Kind::Float);

        //N x K x F
        let activations = softmax_weights.matmul(&peel_layer_state.values);

        activations
    }
}

