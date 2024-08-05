#ifndef STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_ESS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_ESS_HPP

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
#include <iostream>

namespace stan {
namespace analyze {

/**
 * Computes the effective sample size (ESS) for the specified
 * parameter across all chains.  The number of draws per chain must be > 3,
 * and the values across all draws must be finite and not constant.
 * The value returned is the minimum of ESS and (sample_sz * log10(sample_sz).
 * Sample autocovariance is computed using Stan math library implmentation.
 * See https://arxiv.org/abs/1903.08008, section 3.2 for discussion.
 *
 * @param chains matrix of draws across all chains
 * @return effective sample size for the specified parameter
 */
double ess(const Eigen::MatrixXd& chains) {
  const Eigen::Index num_chains = chains.cols();
  const Eigen::Index num_draws = chains.rows();
  Eigen::MatrixXd acov(num_draws, num_chains);
  Eigen::VectorXd chain_mean(num_chains);
  Eigen::VectorXd chain_var(num_chains);

  // compute the per-chain autocovariance
  for (size_t i = 0; i < num_chains; ++i) {
    Eigen::Map<const Eigen::VectorXd> chain_col(chains.col(i).data(),
                                                num_draws);
    Eigen::Map<Eigen::VectorXd> cov_col(acov.col(i).data(), num_draws);
    stan::math::autocovariance<double>(chain_col, cov_col);
    chain_mean(i) = chain_col.mean();
    chain_var(i) = cov_col(0) * num_draws / (num_draws - 1);
  }

  // compute var_plus, eqn (3)
  double w_chain_var = math::mean(chain_var);  // W (within chain var)
  double var_plus = w_chain_var * (num_draws - 1) / num_draws;  // \hat{var}^{+}
  if (num_chains > 1) {
    var_plus += math::variance(chain_mean);  // B (between chain var)
  }

  // Geyer's initial positive sequence, eqn (11)
  Eigen::VectorXd rho_hat_t = Eigen::VectorXd::Zero(num_draws);
  Eigen::VectorXd acov_t(num_chains);
  double rho_hat_even = 1.0;
  rho_hat_t(0) = rho_hat_even;  // lag 0
  double rho_hat_odd = 1 - (w_chain_var - acov.row(1).mean()) / var_plus;
  rho_hat_t(1) = rho_hat_odd;  // lag 1

  // compute autocorrelation at lag t for pair (t, t+1)
  // paired autocorrelation is guaranteed to be positive, monotone and convex
  size_t t = 1;
  while (t < num_draws - 4 && (rho_hat_even + rho_hat_odd > 0)
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

  double num_samples = num_chains * num_draws;
  //  eqn (13): Geyer's truncation rule, w/ modification
  double tau_hat = -1 + 2 * rho_hat_t.head(max_t).sum() + rho_hat_t(max_t + 1);
  // safety check for negative values and with max ess equal to ess*log10(ess)
  tau_hat = std::max(tau_hat, 1 / std::log10(num_samples));
  return (num_samples / tau_hat);
}

/**
 * Computes the split effective sample size (split ESS) using rank based
 * diagnostic for a set of per-chain draws. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * When the number of total draws N is odd, the last draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation

 * @param chains matrix of per-chain draws, num_iters X chain
 * @return potential scale reduction
 */
inline std::pair<double, double> split_rank_normalized_ess(
    const Eigen::MatrixXd& chains) {
  Eigen::MatrixXd split_draws_matrix = split_chains(chains);
  if (!is_finite_and_varies(split_draws_matrix)
      || split_draws_matrix.rows() < 4) {
    return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
  }
  double ess_bulk = ess(rank_transform(split_draws_matrix));
  Eigen::MatrixXd q05 = (split_draws_matrix.array()
                         <= math::quantile(split_draws_matrix.reshaped(), 0.05))
                            .cast<double>();
  double ess_tail_05 = ess(q05);
  Eigen::MatrixXd q95 = (split_draws_matrix.array()
                         >= math::quantile(split_draws_matrix.reshaped(), 0.95))
                            .cast<double>();
  double ess_tail_95 = ess(q95);

  double ess_tail;
  if (std::isnan(ess_tail_05)) {
    ess_tail = ess_tail_95;
  } else if (std::isnan(ess_tail_95)) {
    ess_tail = ess_tail_05;
  } else {
    ess_tail = std::min(ess_tail_05, ess_tail_95);
  }
  return std::make_pair(ess_bulk, ess_tail);
}

}  // namespace analyze
}  // namespace stan

#endif
