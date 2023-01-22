use tch::{nn, kind::Kind, nn::Init, nn::Module, Tensor, 
    nn::Path, nn::Sequential, nn::LinearConfig, IndexOp};

///The peeling state of one layer
#[derive(Debug)]
pub struct PeelLayerState {
    ///Values for look-ups via attentional layers -- N x K x F
    pub values : Tensor,
    ///Attention interaction value -> element linear transforms. One tensor of size N x F x K
    pub interactions : Tensor,
    ///Scaled interaction matrix
    pub scaled_interaction_matrix : Tensor,
}

///Peeling state of a single track of one layer
#[derive(Debug)]
pub struct PeelTrackState {
    ///Value for look-up via subsequent attentional layers -- N x F
    pub value : Tensor,
    ///Attention interaction value -> value weight scalar transform -- N x F
    pub interaction : Tensor
}

///The peeling states of all layers
pub struct PeelLayerStates {
    ///Values for look-ups via attentional layers -- L x N x K x F
    pub values : Tensor,
    ///L x N x F x K
    pub interactions : Tensor,
    ///scaled interaction matrices - L x F x F
    pub scaled_interaction_matrices : Tensor,
}

impl PeelLayerStates {
    pub fn shallow_clone(&self) -> Self {
        Self {
            values : self.values.shallow_clone(),
            interactions : self.interactions.shallow_clone(),
            scaled_interaction_matrices : self.scaled_interaction_matrices.shallow_clone(),
        }
    }
}

///The peeling states of a single track for all layers
pub struct PeelTrackStates {
    ///L x N x F
    pub values : Tensor, 
    ///L x N x F
    pub interactions : Tensor,
}

impl PeelTrackStates {
    pub fn new(peel_track_states : Vec<PeelTrackState>) -> PeelTrackStates {
        let values : Vec<Tensor> = peel_track_states.iter().map(|x| x.value.shallow_clone()).collect();
        let interactions : Vec<Tensor> = peel_track_states.iter().map(|x| x.interaction.shallow_clone()).collect();
        
        let values = Tensor::stack(&values, 0);
        let interactions = Tensor::stack(&interactions, 0);

        PeelTrackStates {
            values,
            interactions
        }
    }
}

impl PeelLayerStates {
    pub fn expand(self, R : usize) -> PeelLayerStates {
        let R = R as i64;
        let values = self.values.expand(&[-1, R, -1, -1], false);
        let interactions = self.interactions.expand(&[-1, R, -1, -1], false);
        PeelLayerStates {
            values,
            interactions,
            scaled_interaction_matrices : self.scaled_interaction_matrices,
        }
    }
    ///Splits to a collection of peel layer states with
    ///the specified number of samples for each
    pub fn split(self, split_sizes : &[i64]) -> Vec<PeelLayerStates> {
        let mut values = self.values.split_with_sizes(split_sizes, 1);
        let mut interactions = self.interactions.split_with_sizes(split_sizes, 1);
        let mut result = Vec::new();
        for (values, interactions) in values.drain(..).zip(interactions.drain(..)) {
            let peel_layer_states = PeelLayerStates {
                values,
                interactions,
                scaled_interaction_matrices : self.scaled_interaction_matrices.shallow_clone(),
            };
            result.push(peel_layer_states);
        }
        result
    }
    ///Merges samples of all peel layer states
    pub fn merge(mut peel_layer_states : Vec<PeelLayerStates>) -> PeelLayerStates {
        let values : Vec<Tensor> = peel_layer_states.iter().map(|x| x.values.shallow_clone()).collect();
        let interactions : Vec<Tensor> = peel_layer_states.iter().map(|x| x.interactions.shallow_clone()).collect();

        let values = Tensor::concat(&values, 1);
        let interactions = Tensor::concat(&interactions, 1);

        //Scaled interaction matrices shouldn't really change
        let scaled_interaction_matrices = peel_layer_states.pop().unwrap().scaled_interaction_matrices;
        PeelLayerStates {
            values,
            interactions,
            scaled_interaction_matrices,
        }
    }
    ///Gets the peeling states of a given layer
    pub fn get_layer_state(&self, layer_num : usize) -> PeelLayerState {
        let layer_num = layer_num as i64;
        let values = self.values.i((layer_num, .., .., ..));
        let interactions = self.interactions.i((layer_num, .., .., ..));
        let scaled_interaction_matrix = self.scaled_interaction_matrices.i((layer_num, .., ..));
        PeelLayerState {
            values,
            interactions,
            scaled_interaction_matrix
        }
    }
    ///Stack together per-layer PeelLayerStates into a PeelLayerStates
    ///for all layers.
    pub fn new(peel_layer_states : Vec<PeelLayerState>) -> PeelLayerStates {
        let values : Vec<Tensor> = peel_layer_states.iter().map(|x| x.values.shallow_clone()).collect();         
        let interactions : Vec<Tensor> = peel_layer_states.iter().map(|x| x.interactions.shallow_clone()).collect();
        let scaled_interaction_matrices : Vec<Tensor> = peel_layer_states.iter().map(|x| x.scaled_interaction_matrix.shallow_clone()).collect();

        let values = Tensor::stack(&values, 0);
        let interactions = Tensor::stack(&interactions, 0);
        let scaled_interaction_matrices = Tensor::stack(&scaled_interaction_matrices, 0);

        PeelLayerStates {
            values,
            interactions,
            scaled_interaction_matrices
        }
    }
    ///Pushes track along every layer to derive a new peeling state for every layer.
    pub fn push_tracks(self, peel_track_states : &PeelTrackStates) -> PeelLayerStates {
        let track_values = peel_track_states.values.unsqueeze(2);
        let track_interactions = peel_track_states.interactions.unsqueeze(3);

        let values = Tensor::concat(&[&self.values, &track_values], 2);
        let interactions = Tensor::concat(&[&self.interactions, &track_interactions], 3);

        PeelLayerStates {
            values,
            interactions,
            ..self
        }
    }
}
