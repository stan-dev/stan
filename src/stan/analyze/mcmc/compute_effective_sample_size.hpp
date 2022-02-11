#ifndef STAN_ANALYZE_MCMC_COMPUTE_EFFECTIVE_SAMPLE_SIZE_HPP
#define STAN_ANALYZE_MCMC_COMPUTE_EFFECTIVE_SAMPLE_SIZE_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {
/**
 * Computes the effective sample size (ESS) for the specified
 * parameter across all kept samples.  The value returned is the
 * minimum of ESS and the number_total_draws *
 * log10(number_total_draws).
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Note that the effective sample size
 * can not be estimated with less than four draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return effective sample size for the specified parameter
 */
inline double compute_effective_sample_size(std::vector<const double*> draws,
                                            std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  if (num_draws < 4) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // check if chains are constant; all equal to first draw's value
  bool are_all_const = false;
  Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

  for (int chain_idx = 0; chain_idx < num_chains; chain_idx++) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain_idx], sizes[chain_idx]);

    for (int n = 0; n < num_draws; n++) {
      if (!std::isfinite(draw(n))) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    init_draw(chain_idx) = draw(0);

    if (draw.isApproxToConstant(draw(0))) {
      are_all_const |= true;
    }
  }

  if (are_all_const) {
    // If all chains are constant then return NaN
    // if they all equal the same constant value
    if (init_draw.isApproxToConstant(init_draw(0))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  Eigen::Matrix<Eigen::VectorXd, Eigen::Dynamic, 1> acov(num_chains);
  Eigen::VectorXd chain_mean(num_chains);
  Eigen::VectorXd chain_var(num_chains);
  for (int chain = 0; chain < num_chains; ++chain) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain], sizes[chain]);
    autocovariance<double>(draw, acov(chain));
    chain_mean(chain) = draw.mean();
    chain_var(chain) = acov(chain)(0) * num_draws / (num_draws - 1);
  }

  double mean_var = chain_var.mean();
  double var_plus = mean_var * (num_draws - 1) / num_draws;
  if (num_chains > 1)
    var_plus += math::variance(chain_mean);
  Eigen::VectorXd rho_hat_s(num_draws);
  rho_hat_s.setZero();
  Eigen::VectorXd acov_s(num_chains);
  for (int chain = 0; chain < num_chains; ++chain)
    acov_s(chain) = acov(chain)(1);
  double rho_hat_even = 1.0;
  rho_hat_s(0) = rho_hat_even;
  double rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
  rho_hat_s(1) = rho_hat_odd;

  // Convert raw autocovariance estimators into Geyer's initial
  // positive sequence. Loop only until num_draws - 4 to
  // leave the last pair of autocorrelations as a bias term that
  // reduces variance in the case of antithetical chains.
  size_t s = 1;
  while (s < (num_draws - 4) && (rho_hat_even + rho_hat_odd) > 0) {
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(s + 1);
    rho_hat_even = 1 - (mean_var - acov_s.mean()) / var_plus;
    for (int chain = 0; chain < num_chains; ++chain)
      acov_s(chain) = acov(chain)(s + 2);
    rho_hat_odd = 1 - (mean_var - acov_s.mean()) / var_plus;
    if ((rho_hat_even + rho_hat_odd) >= 0) {
      rho_hat_s(s + 1) = rho_hat_even;
      rho_hat_s(s + 2) = rho_hat_odd;
    }
    s += 2;
  }

  int max_s = s;
  // this is used in the improved estimate, which reduces variance
  // in antithetic case -- see tau_hat below
  if (rho_hat_even > 0)
    rho_hat_s(max_s + 1) = rho_hat_even;

  // Convert Geyer's initial positive sequence into an initial
  // monotone sequence
  for (int s = 1; s <= max_s - 3; s += 2) {
    if (rho_hat_s(s + 1) + rho_hat_s(s + 2) > rho_hat_s(s - 1) + rho_hat_s(s)) {
      rho_hat_s(s + 1) = (rho_hat_s(s - 1) + rho_hat_s(s)) / 2;
      rho_hat_s(s + 2) = rho_hat_s(s + 1);
    }
  }

  double num_total_draws = num_chains * num_draws;
  // Geyer's truncated estimator for the asymptotic variance
  // Improved estimate reduces variance in antithetic case
  double tau_hat = -1 + 2 * rho_hat_s.head(max_s).sum() + rho_hat_s(max_s + 1);
  return std::min(num_total_draws / tau_hat,
                  num_total_draws * std::log10(num_total_draws));
}

/**
 * Computes the effective sample size (ESS) for the specified
 * parameter across all kept samples.  The value returned is the
 * minimum of ESS and the number_total_draws *
 * log10(number_total_draws).
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Note that the effective sample size
 * can not be estimated with less than four draws.  Argument size
 * will be broadcast to same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param size size of chains
 * @return effective sample size for the specified parameter
 */
inline double compute_effective_sample_size(std::vector<const double*> draws,
                                            size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_effective_sample_size(draws, sizes);
}

/**
 * Computes the split effective sample size (ESS) for the specified
 * parameter across all kept samples.  The value returned is the
 * minimum of ESS and the number_total_draws *
 * log10(number_total_draws). When the number of total draws N is
 * odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Note that the effective sample size
 * can not be estimated with less than four draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return effective sample size for the specified parameter
 */
inline double compute_split_effective_sample_size(
    std::vector<const double*> draws, std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  std::vector<const double*> split_draws = split_chains(draws, sizes);

  double half = num_draws / 2.0;
  std::vector<size_t> half_sizes(2 * num_chains, std::floor(half));

  return compute_effective_sample_size(split_draws, half_sizes);
}

/**
 * Computes the split effective sample size (ESS) for the specified
 * parameter across all kept samples.  The value returned is the
 * minimum of ESS and the number_total_draws *
 * log10(number_total_draws). When the number of total draws N is
 * odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Note that the effective sample size
 * can not be estimated with less than four draws.  Argument size
 * will be broadcast to same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param size size of chains
 * @return effective sample size for the specified parameter
 */
inline double compute_split_effective_sample_size(
    std::vector<const double*> draws, size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_split_effective_sample_size(draws, sizes);
}

}  // namespace analyze
}  // namespace stan

#endif
