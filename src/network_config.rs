use tch::{nn, nn::Init, nn::Module, Tensor, nn::Path, nn::Sequential};
use crate::network::*;
use crate::neural_utils::*;

///M : matrix (flattened) dims
///F : feature dims
pub struct NetworkConfig {
    ///M x M -> F (single)
    injector_net : ConcatThenSequential,
    ///F x F -> F (combined)
    combiner_net : ConcatThenSequential,
    ///F -> Scalar
    value_extraction_net : Sequential,
    ///F (combined) x F (single) -> 2F
    left_policy_extraction_net : ConcatThenSequential,
    ///F (combined) x F (single) -> 2F
    right_policy_extraction_net : ConcatThenSequential
}

impl NetworkConfig {
    pub fn new(vs : &nn::Path) -> NetworkConfig {
        let injector_net = injector_net(vs / "injector");
        let combiner_net = combiner_net(vs / "combiner");
        let value_extraction_net = value_extraction_net(vs / "value_extractor");
        let left_policy_extraction_net = half_policy_extraction_net(vs / "left_policy_vector_supplier");
        let right_policy_extraction_net = half_policy_extraction_net(vs / "right_policy_vector_supplier");
        NetworkConfig {
            injector_net,
            combiner_net,
            value_extraction_net,
            left_policy_extraction_net,
            right_policy_extraction_net
        }
    }
    ///Given embeddings of dimension F, yields a combined embedding of dimension F
    fn combine_embeddings(&self, embeddings : &[Tensor]) -> Tensor {
        let k = embeddings.len();
        if k == 1 {
            embeddings[0].shallow_clone()
        } else {
            let mut updated = Vec::new();
            for i in 0..k {
                let elem = embeddings[i].shallow_clone();
                if i % 2 == 0 {
                    updated.push(elem);
                } else {
                    let prev_elem = updated.pop().unwrap();
                    let combined = self.combiner_net.forward(&prev_elem, &elem);
                    updated.push(combined);
                }
            }
            self.combine_embeddings(&updated)
        }
    }

    ///Given a collection of K [flattened] matrices (dimension M), and a flattened matrix target
    ///(dimension M), yields a computed (Value [Scalar], Policy [Matrix of dimension KxK]) pair
    pub fn get_value_and_policy(&self, flattened_matrix_set : &[Tensor], flattened_matrix_target : &Tensor) ->
                 (Tensor, Tensor) {

        //First, compute the embeddings for every input matrix
        let k = flattened_matrix_set.len();
        let mut single_embeddings = Vec::new();
        for i in 0..k {
            let flattened_matrix = &flattened_matrix_set[i];
            let embedding = self.injector_net.forward(flattened_matrix, flattened_matrix_target);
            single_embeddings.push(embedding);
        }
        //Then, combine embeddings repeatedly until there's only one "master" embedding for the
        //entire collection of matrices
        let combined_embedding = self.combine_embeddings(&single_embeddings);

        //Now that we have the combined embedding, extract the value
        let value = self.value_extraction_net.forward(&combined_embedding);

        //Compute left and right policy vectors
        let mut left_policy_vecs = Vec::new();
        let mut right_policy_vecs = Vec::new();
        for i in 0..k {
            let single_embedding = &single_embeddings[i];
            let left_policy_vec = self.left_policy_extraction_net.forward(&combined_embedding, single_embedding);
            let right_policy_vec = self.right_policy_extraction_net.forward(&combined_embedding, single_embedding);
            left_policy_vecs.push(left_policy_vec);
            right_policy_vecs.push(right_policy_vec);
        }

        //Combine the left and right policy vectors by doing pairwise dot-products
        let left_policy_mat = Tensor::stack(&left_policy_vecs, 0);
        let right_policy_mat = Tensor::stack(&right_policy_vecs, 0);
        let policy_mat = left_policy_mat.dot(&right_policy_mat.tr());

        (value, policy_mat)
    }
}
