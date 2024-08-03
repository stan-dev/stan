#ifndef STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_RHAT_HPP
#define STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_RHAT_HPP

#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/analyze/mcmc/rank_normalization.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes square root of marginal posterior variance of the estimand by the
 * weighted average of within-chain variance W and between-chain variance B.
 *
 * @param chains stores chains in columns
 * @return square root of ((N-1)/N)W + B/N
 */
inline double rhat(const Eigen::MatrixXd& chains) {
  const Eigen::Index num_chains = chains.cols();
  const Eigen::Index num_draws = chains.rows();

  Eigen::RowVectorXd within_chain_means = chains.colwise().mean();
  double across_chain_mean = within_chain_means.mean();
  double between_variance
      = num_draws
        * (within_chain_means.array() - across_chain_mean).square().sum()
        / (num_chains - 1);
  double within_variance =
      // Divide each row by chains and get sum of squares for each chain
      // (getting a vector back)
      ((chains.rowwise() - within_chain_means)
           .array()
           .square()
           .colwise()
           // divide each sum of square by num_draws, sum the sum of squares,
           // and divide by num chains
           .sum()
       / (num_draws - 1.0))
          .sum()
      / num_chains;

  return sqrt((between_variance / within_variance + num_draws - 1) / num_draws);
}

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
