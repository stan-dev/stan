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

  double sample_var
      = (chains.array() - chains.mean()).square().sum() / (chains.size() - 1);
  return std::sqrt(sample_var / ess(chains));
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

  // center the data, take abs value
  Eigen::MatrixXd draws_ctr = (chains.array() - chains.mean()).abs().matrix();

  // posterior pkg fn `ess_mean` computes on split chains
  double ess_mean = ess(split_chains(draws_ctr));

  // estimated variance (2nd moment)
  double Evar = draws_ctr.array().square().mean();

  // variance of variance, adjusted for ESS
  double fourth_moment = draws_ctr.array().pow(4).mean();
  double varvar = (fourth_moment - std::pow(Evar, 2)) / ess_mean;

  // variance of standard deviation - use Taylor series approximation
  double varsd = varvar / Evar / 4.0;
  return std::sqrt(varsd);
}

}  // namespace analyze
}  // namespace stan

#endif
