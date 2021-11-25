use rand::Rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Gamma, Normal};

#[derive(Clone, Copy)]
pub struct NormalInverseChiSquared {
    pub mean : f64,
    pub mean_observations : i64,
    pub variance_observations : i64,
    pub variance : f64
}

impl NormalInverseChiSquared {
    pub fn sample<R : RngCore + ?Sized>(&self, rng : &mut R) -> f64 {
        let alpha = self.variance_observations as f64 * 0.5f64;
        let beta = alpha * self.variance;
        let theta = 1.0f64 / beta;
        let gamma = Gamma::new(alpha, theta).unwrap();
        let gamma_sample = gamma.sample(rng);
        let sampled_variance = 1.0f64 / gamma_sample;

        let normal_variance = sampled_variance / (self.mean_observations as f64);
        let normal_std_dev = normal_variance.sqrt();
        let normal = Normal::new(self.mean, normal_std_dev).unwrap();

        normal.sample(rng)
    }

    pub fn uninformative() -> Self {
        NormalInverseChiSquared {
            mean : 0.0f64,
            mean_observations : 0,
            variance_observations : -1,
            variance : 0.0f64
        }
    }
    pub fn as_single_observation(&self) -> Self {
        let mean = self.mean;
        let mean_observations = 1;
        let variance_observations = 1;
        let variance = self.variance;
        NormalInverseChiSquared {
            mean,
            mean_observations,
            variance_observations,
            variance
        }
    }
    pub fn invert(self) -> Self {
        let mean = self.mean;
        let mean_observations = -self.mean_observations;
        let variance_observations = -self.variance_observations;
        let variance = self.variance;
        NormalInverseChiSquared {
            mean,
            mean_observations,
            variance_observations,
            variance
        }
    }
    pub fn merge(self, other : &Self) -> Self {
        let mean_observations = self.mean_observations + other.mean_observations;
        let recip_mean_observations = 1.0f64 / (mean_observations as f64);

        let mean = (self.mean * self.mean_observations as f64 + other.mean * other.mean_observations as f64)
                   * recip_mean_observations;
        
        let variance_observations = self.variance_observations + other.variance_observations; 
        let recip_variance_observations = 1.0f64 / (variance_observations as f64);

        let my_total_variance = self.variance_observations as f64 * self.variance;
        let other_total_variance = other.variance_observations as f64 * other.variance;

        let mean_observations_product = self.mean_observations * other.mean_observations;
        let interaction_scaling_factor = mean_observations_product as f64 * recip_mean_observations;

        let mean_diff = self.mean - other.mean;
        let sq_diff = mean_diff * mean_diff;

        let interaction_variance = interaction_scaling_factor * sq_diff;

        let total_variance = my_total_variance + other_total_variance + interaction_variance;
        let variance = total_variance * recip_variance_observations;

        NormalInverseChiSquared {
            mean,
            mean_observations,
            variance_observations,
            variance
        }
    }
    pub fn update(self, value : f64) -> Self {
        //Based on
        //https://en.wikipedia.org/wiki/Normal_distribution#Bayesian_analysis_of_the_normal_distribution
        if (self.mean_observations == 0 && self.variance_observations == -1) {
            NormalInverseChiSquared {
                mean : value,
                mean_observations : 1,
                variance_observations : 0,
                variance : 0.0f64
            }
        } else {
            let mean_observations = self.mean_observations + 1;
            let recip_mean_observations = 1.0f64 / (mean_observations as f64);

            let mean = (self.mean_observations as f64 * self.mean + value) * recip_mean_observations;

            let diff_from_prior_mean = value - self.mean;
            let sq_diff = diff_from_prior_mean * diff_from_prior_mean;

            let variance_observations = self.variance_observations + 1; 

            let recip_variance_observations = 1.0f64 / (variance_observations as f64);

            let new_variance_fraction = self.mean_observations as f64 * recip_mean_observations;
            let total_variance = (self.variance_observations as f64 * self.variance) + new_variance_fraction * sq_diff;
            let variance = total_variance * recip_variance_observations;

            NormalInverseChiSquared {
                mean,
                mean_observations,
                variance_observations,
                variance
            }
        }
    }
}
