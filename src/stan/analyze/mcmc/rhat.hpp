#ifndef STAN_ANALYZE_MCMC_RHAT_HPP
#define STAN_ANALYZE_MCMC_RHAT_HPP

#include <stan/math/prim.hpp>
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

}  // namespace analyze
}  // namespace stan

#endif
