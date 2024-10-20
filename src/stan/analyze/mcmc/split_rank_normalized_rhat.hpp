#ifndef STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_RHAT_HPP
#define STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_RHAT_HPP

#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/analyze/mcmc/rank_normalization.hpp>
#include <stan/analyze/mcmc/rhat.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <stan/math/prim.hpp>
#include <limits>
#include <utility>

namespace stan {
namespace analyze {

/**
 * Computes the split potential scale reduction (split Rhat) using rank based
 * diagnostic for a set of per-chain draws. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * When the number of total draws N is odd, the last draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation

 * @param chains matrix of per-chain samples, num_iters X chain
 * @return potential scale reduction for the specified parameter
 */
inline std::pair<double, double> split_rank_normalized_rhat(
    const Eigen::MatrixXd& chains) {
  Eigen::MatrixXd split_draws_matrix = split_chains(chains);
  if (!is_finite_and_varies(split_draws_matrix)) {
    return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
  }
  double rhat_bulk = rhat(rank_transform(split_draws_matrix));
  // zero-center the draws at the median
  double rhat_tail = rhat(
      rank_transform((split_draws_matrix.array()
                      - math::quantile(split_draws_matrix.reshaped(), 0.5))
                         .abs()));
  return std::make_pair(rhat_bulk, rhat_tail);
}

}  // namespace analyze
}  // namespace stan

#endif
