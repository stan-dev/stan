#ifndef STAN_ANALYZE_MCMC_MCSE_HPP
#define STAN_ANALYZE_MCMC_MCSE_HPP

#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <stan/analyze/mcmc/ess.hpp>
#include <stan/math/prim.hpp>
#include <cmath>
#include <limits>
#include <utility>

namespace stan {
namespace analyze {

/**
 * Computes the Monte Carlo standard error of the mean.
 * See https://arxiv.org/abs/1903.08008, section 4.4.
 * Follows implementation in the R posterior package.
 *
 * @param chains matrix of draws across all chains
 * @return mcse
 */
inline double mcse_mean(const Eigen::MatrixXd& chains) {
  if (chains.rows() < 8 || !is_finite_and_varies(chains))
    return std::numeric_limits<double>::quiet_NaN();

  double sample_var
      = (chains.array() - chains.mean()).square().sum() / (chains.size() - 1);
  // posterior pkg fn `ess_mean` computes on split chains
  return std::sqrt(sample_var / ess(split_chains(chains)));
}

/**
 * Computes the Monte Carlo standard error of the standard deviation.
 * See https://arxiv.org/abs/1903.08008, section 4.4.
 * Follows implementation in the R posterior package.  The effective
 * sample size is that of the squared deviations from the mean.
 *
 * @param chains matrix of draws across all chains
 * @return mcse
 */
inline double mcse_sd(const Eigen::MatrixXd& chains) {
  if (chains.rows() < 8 || !is_finite_and_varies(chains)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // squared deviations from the mean
  Eigen::MatrixXd draws_sq = (chains.array() - chains.mean()).square().matrix();

  // posterior pkg fn `ess_mean` computes on split chains
  double ess_mean = ess(split_chains(draws_sq));

  // estimated variance (2nd moment)
  double Evar = draws_sq.mean();

  // variance of variance, adjusted for ESS
  double varvar = (draws_sq.array() - Evar).square().mean() / ess_mean;

  // variance of standard deviation - use Taylor series approximation
  double varsd = varvar / Evar / 4.0;
  return std::sqrt(varsd);
}

}  // namespace analyze
}  // namespace stan

#endif
