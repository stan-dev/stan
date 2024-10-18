#ifndef STAN_ANALYZE_MCMC_MCSE_HPP
#define STAN_ANALYZE_MCMC_MCSE_HPP

#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/math/prim.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

namespace stan {
namespace analyze {

/**
 * Computes the mean Monte Carlo error estimate for the central 90% interval.
 * See https://arxiv.org/abs/1903.08008, section 4.4.
 * Follows implementation in the R posterior package.
 *
 * @param chains matrix of draws across all chains
 * @return mcse
 */
inline double mcse_mean(const Eigen::MatrixXd& chains) {
  const Eigen::Index num_draws = chains.rows();
  if (chains.rows() < 4 || !is_finite_and_varies(chains))
    return std::numeric_limits<double>::quiet_NaN();

  double sd
      = (chains.array() - chains.mean()).square().sum() / (chains.size() - 1);
  return std::sqrt(sd / ess(chains));
}

/**
 * Computes the standard deviation of the Monte Carlo error estimate
 * https://arxiv.org/abs/1903.08008, section 4.4.
 * Follows implementation in the R posterior package:
 * https://github.com/stan-dev/posterior/blob/98bf52329d68f3307ac4ecaaea659276ee1de8df/R/convergence.R#L478-L496
 *
 * @param chains matrix of draws across all chains
 * @return mcse
 */
inline double mcse_sd(const Eigen::MatrixXd& chains) {
  if (chains.rows() < 4 || !is_finite_and_varies(chains))
    return std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd diffs = (chains.array() - chains.mean()).matrix();
  double Evar = diffs.array().square().mean();
  double varvar = (math::mean(diffs.array().pow(4) - Evar * Evar))
                  / ess(diffs.array().abs().matrix());
  return std::sqrt(varvar / Evar / 4);
}

}  // namespace analyze
}  // namespace stan

#endif
