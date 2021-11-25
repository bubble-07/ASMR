pub const FLATTENED_MATRIX_DIM : i64 = 16;
pub const NUM_FEAT_MAPS : i64 = 32;
pub const SINGLETON_INJECTION_LAYERS : i64 = 20;
pub const COMBINING_LAYERS : i64 = 40;
pub const POLICY_EXTRACTION_LAYERS : i64 = 5;

//Data generation parameters will likely be
//-dims of matrix
//-deviation parameter of the log-normal distribution [used to generate matrix entries
//   centered at zero, with randomized positive/negative flipping]
//-number of matrices in the initial set (likely small relative to the dimension)
//-number of turns (probably small, roughly geometrically-distributed, always >= 2)


//Random notes: Rubik's cube group, for instance, can have representations with matrix dimension ~20,
//and ~6 generators. 20 turns ["God's number"] is the upper bound on the minimal number of terms.
//This would be useful for a particularly inspired set of choices of parameters,
//and from this, we could also potentially _measure_ performance on solving Rubik's cubes
//as a benchmark [bearing in mind that we're solving a different, more general problem]
//
//For the number of moves, though, it's important to bear in mind that our "moves"
//are multiplication of _two_ matrices. If we take the naive assumption that
//repeated squarings would suffice, then our reference number of moves is
//log_2(20) which is about 4.3219.
//So, setting the half-life of a geometric distribution for # moves to happen around 4 seems reasonable
