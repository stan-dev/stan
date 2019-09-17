#ifndef STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP
#define STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP

#include <stan/math/prim/mat.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <stan/analyze/mcmc/welford_variance.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes the potential scale reduction (Rhat) for the specified
 * parameter across all kept samples.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_potential_scale_reduction(
    std::vector<const double*> draws, std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  // check if chains are constant; all equal to first draw's value
  Eigen::VectorXd draw_val(num_chains);
  for (int chain = 0; chain < num_chains; chain++)
    draw_val(chain) = static_cast<double>(chain);

  for (int chain = 0; chain < num_chains; chain++) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain], sizes[chain]);

    for (int n = 0; n < num_draws; n++) {
      if (!boost::math::isfinite(draw(n))) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    if (draw.isApproxToConstant(draw(0))) {
      draw_val(chain) = draw(0);
    }
  }

  if (draw_val.isApproxToConstant(draw_val(0))) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  Eigen::VectorXd chain_mean(num_chains);
  Eigen::VectorXd chain_var(num_chains);

  for (int chain = 0; chain < num_chains; ++chain) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain], sizes[chain]);
    chain_mean(chain) = draw.mean();
    chain_var(chain) = welford_variance(draw);
  }

  double var_between = num_draws * welford_variance(chain_mean);
  double var_within = chain_var.mean();

  // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
  return sqrt((var_between / var_within + num_draws - 1) / num_draws);
}

/**
 * Computes the potential scale reduction (Rhat) for the specified
 * parameter across all kept samples.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Argument size will be broadcast to
 * same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_potential_scale_reduction(
    std::vector<const double*> draws, size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_potential_scale_reduction(draws, sizes);
}

/**
 * Computes the split potential scale reduction (Rhat) for the
 * specified parameter across all kept samples.  When the number of
 * total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_split_potential_scale_reduction(
    std::vector<const double*> draws, std::vector<size_t> sizes) {
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

/**
 * Computes the split potential scale reduction (Rhat) for the
 * specified parameter across all kept samples.  When the number of
 * total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Argument size will be broadcast to
 * same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_split_potential_scale_reduction(
    std::vector<const double*> draws, size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_split_potential_scale_reduction(draws, sizes);
}

}  // namespace analyze
}  // namespace stan

#endif
