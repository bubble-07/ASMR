use rand::Rng;
use std::ops::Range;
use std::iter;
use std::iter::Iterator;

///A split in a sequence, logically separating it into two different
///parts: a "training data" part consisting only of full-batches,
///and a "validation data" part consisting of the part of the
///sequence which is not training data. 
#[derive(Clone, Copy)]
pub struct BatchSplit {
    batch_size : usize,
    full_size : usize,
    num_training_batches : usize,
}

impl BatchSplit {
    ///Grabs a random training batch. Returns None if there's not enough
    ///samples for there to be any training batches.
    pub fn grab_training_batch<R : Rng + ?Sized>(&self, rng : &mut R) -> Option<Range<i64>> {
        if self.num_training_batches == 0 {
            Option::None
        } else {
            let batch_index = rng.gen_range(0..self.num_training_batches);
            let batch_start = batch_index * self.batch_size;
            let batch_end = batch_start + self.batch_size;
            Option::Some((batch_start as i64)..(batch_end as i64))
        }
    }

    ///Returns an iterator over validation batches, in terms of
    ///(weight, index range) pairs. The weights in the returned
    ///iterator sum to 1.0.
    pub fn iter_validation_batches<'a>(&'a self) -> impl Iterator<Item = (f64, Range<i64>)> + 'a {
        let validation_start = self.batch_size * self.num_training_batches;
        let num_validation_elements = self.full_size - validation_start;
        let num_validation_batches = num_validation_elements / self.batch_size;
        let leftover_validation_elements = num_validation_elements - num_validation_batches * self.batch_size;

        let ordinary_weight = (self.batch_size as f64) / (num_validation_elements as f64);
        let leftover_weight = (leftover_validation_elements as f64) / (num_validation_elements as f64);

        let full_batches = (0..num_validation_batches)
                           .map(move |batch_num| {
                               let start_index = validation_start + (batch_num * self.batch_size);
                               let end_index = start_index + self.batch_size;
                               (ordinary_weight, (start_index as i64)..(end_index as i64))
                           });

        let mut leftover_iter = None;
        if leftover_validation_elements > 0 {
            let total_batches = self.num_training_batches + num_validation_batches;
            let start_index = total_batches * self.batch_size;
            let leftover_range = (start_index as i64)..(self.full_size as i64);
            let leftover_pair = (leftover_weight, leftover_range);
            leftover_iter = Option::Some(iter::once(leftover_pair));
        }
        full_batches.chain(leftover_iter.into_iter().flatten())
    }

    ///Constructs a new BatchSplit from the size of the full sequence,
    ///the size of a batch, and the recommended minimum number of validation
    ///batches. If there are insufficient elements in the whole sequence to
    ///accomodate the desired number of validation batches plus one, then _all_
    ///elements in the sequence will wind up being allocated to the validation
    ///partition. Otherwise, as many full-batch-size batches as possible
    ///will be allocated towards the training data, with leftover elements
    ///falling into the validation batch portion
    pub fn new(full_size : usize, batch_size : usize, 
               recommended_min_validation_batches : usize) -> BatchSplit {
        let maximal_full_batches = full_size / batch_size;
        let num_training_batches = 
            if (maximal_full_batches <= recommended_min_validation_batches) {
                0
            }
            else {
                maximal_full_batches - recommended_min_validation_batches
            };

        BatchSplit {
            batch_size,
            full_size,
            num_training_batches,
        }
    }
}
