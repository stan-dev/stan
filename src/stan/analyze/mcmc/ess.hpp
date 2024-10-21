#ifndef STAN_ANALYZE_MCMC_ESS_HPP
#define STAN_ANALYZE_MCMC_ESS_HPP

#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes the effective sample size (ESS) for the specified
 * parameter across all chains.  The number of draws per chain must be > 3,
 * and the values across all draws must be finite and not constant.
 * See https://arxiv.org/abs/1903.08008, section 3.2 for discussion.
 *
 * Sample autocovariance is computed using the implementation in this namespace
 * which normalizes lag-k autocorrelation estimators by N instead of (N - k),
 * yielding biased but more stable estimators as discussed in Geyer (1992); see
 * https://projecteuclid.org/euclid.ss/1177011137.
 *
 * @param chains matrix of draws across all chains
 * @return effective sample size for the specified parameter
 */
double ess(const Eigen::MatrixXd& chains) {
  const Eigen::Index num_chains = chains.cols();
  const Eigen::Index draws_per_chain = chains.rows();
  Eigen::MatrixXd acov(draws_per_chain, num_chains);
  Eigen::VectorXd chain_mean(num_chains);
  Eigen::VectorXd chain_var(num_chains);

  // compute the per-chain autocovariance
  for (size_t i = 0; i < num_chains; ++i) {
    chain_mean(i) = chains.col(i).mean();
    Eigen::Map<const Eigen::VectorXd> draw_col(chains.col(i).data(),
                                               draws_per_chain);
    Eigen::VectorXd cov_col(draws_per_chain);
    autocovariance<double>(draw_col, cov_col);
    acov.col(i) = cov_col;
    chain_var(i) = cov_col(0) * draws_per_chain / (draws_per_chain - 1);
  }

  // compute var_plus, eqn (3)
  double w_chain_var = math::mean(chain_var);  // W (within chain var)
  double var_plus
      = w_chain_var * (draws_per_chain - 1) / draws_per_chain;  // \hat{var}^{+}
  if (num_chains > 1) {
    var_plus += math::variance(chain_mean);  // B (between chain var)
  }

  // Geyer's initial positive sequence, eqn (11)
  Eigen::VectorXd rho_hat_t = Eigen::VectorXd::Zero(draws_per_chain);
  double rho_hat_even = 1.0;
  rho_hat_t(0) = rho_hat_even;  // lag 0

  Eigen::VectorXd acov_t(num_chains);
  for (size_t i = 0; i < num_chains; ++i) {
    acov_t(i) = acov(1, i);
  }
  double rho_hat_odd = 1 - (w_chain_var - acov_t.mean()) / var_plus;
  rho_hat_t(1) = rho_hat_odd;  // lag 1

  // compute autocorrelation at lag t for pair (t, t+1)
  // paired autocorrelation is guaranteed to be positive, monotone and convex
  size_t t = 1;
  while (t < draws_per_chain - 4 && (rho_hat_even + rho_hat_odd > 0)
         && !std::isnan(rho_hat_even + rho_hat_odd)) {
    for (size_t i = 0; i < num_chains; ++i) {
      acov_t(i) = acov.col(i)(t + 1);
    }
    rho_hat_even = 1 - (w_chain_var - acov_t.mean()) / var_plus;
    for (size_t i = 0; i < num_chains; ++i) {
      acov_t(i) = acov.col(i)(t + 2);
    }
    rho_hat_odd = 1 - (w_chain_var - acov_t.mean()) / var_plus;
    if ((rho_hat_even + rho_hat_odd) >= 0) {
      rho_hat_t(t + 1) = rho_hat_even;
      rho_hat_t(t + 2) = rho_hat_odd;
    }
    // convert initial positive sequence into an initial monotone sequence
    if (rho_hat_t(t + 1) + rho_hat_t(t + 2) > rho_hat_t(t - 1) + rho_hat_t(t)) {
      rho_hat_t(t + 1) = (rho_hat_t(t - 1) + rho_hat_t(t)) / 2;
      rho_hat_t(t + 2) = rho_hat_t(t + 1);
    }
    t += 2;
  }

  auto max_t = t;  // max lag, used for truncation
  //  see discussion p. 8, par "In extreme antithetic cases, "
  if (rho_hat_even > 0) {
    rho_hat_t(max_t + 1) = rho_hat_even;
  }

  double draws_total = num_chains * draws_per_chain;
  //  eqn (13): Geyer's truncation rule, w/ modification
  double tau_hat = -1 + 2 * rho_hat_t.head(max_t).sum() + rho_hat_t(max_t + 1);
  // safety check for negative values and with max ess equal to ess*log10(ess)
  tau_hat = std::max(tau_hat, 1 / std::log10(draws_total));
  return (draws_total / tau_hat);
}

}  // namespace analyze
}  // namespace stan

#endif
