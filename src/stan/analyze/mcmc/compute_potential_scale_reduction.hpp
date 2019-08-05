#ifndef STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP
#define STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

namespace stan {
namespace analyze {

  /**
   * Computes the potential scale reduction (Rhat) for the specified
   * parameter across all kept samples.
   *
   * Current implementation assumes chains are all of equal size and
   * draws are stored in contiguous blocks of memory.
   *
   * @param draws stores pointers to arrays of chains
   * @param sizes stores sizes of chains
   *
   * @return potential scale reduction for the specified parameter
   */
  inline
  double compute_potential_scale_reduction(std::vector<const double*> draws,
                                           std::vector<size_t> sizes) {
    int num_chains = sizes.size();
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    Eigen::VectorXd chain_mean(num_chains);
    Eigen::VectorXd chain_var(num_chains);

    for (int chain = 0; chain < num_chains; chain++) {
      Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>>
        draw(draws[chain], sizes[chain]);
      chain_mean(chain) = draw.mean();
      chain_var(chain) = (draw.array() - chain_mean(chain)).square().sum()
        / (num_draws - 1);
    }

    double var_between = num_draws *
      (chain_mean.array() - chain_mean.mean()).square().sum() / (num_chains - 1);
    double var_within = chain_var.mean();

    // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
    return sqrt( (var_between / var_within + num_draws - 1) / num_draws );
  }

  inline
  double compute_potential_scale_reduction(std::vector<const double*> draws,
                                           size_t size) {
    int num_chains = draws.size();
    std::vector<size_t> sizes(num_chains, size);
    return compute_potential_scale_reduction(draws, sizes);
  }

  inline
  double compute_split_potential_scale_reduction(std::vector<const double*> draws,
                                                 std::vector<size_t> sizes) {
    int num_chains = sizes.size();
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    std::vector<const double*> split_draws = split_chains(draws, sizes);

    double half = num_draws / 2.0;
    std::vector<size_t> half_sizes(2 * num_chains, std::floor(half));

    return compute_potential_scale_reduction(split_draws, half_sizes);
  }

  inline
  double compute_split_potential_scale_reduction(std::vector<const double*> draws,
                                                 size_t size) {
    int num_chains = draws.size();
    std::vector<size_t> sizes(num_chains, size);
    return compute_split_potential_scale_reduction(draws, sizes);
  }


}  // namespace analyze
}  // namespace stan

#endif
