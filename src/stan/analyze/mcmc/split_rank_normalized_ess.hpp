#ifndef STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_ESS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_RANK_NORMALIZED_ESS_HPP

#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/ess.hpp>
#include <stan/analyze/mcmc/check_chains.hpp>
#include <stan/analyze/mcmc/rank_normalization.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes the split effective sample size (split ESS) using rank based
 * diagnostic for a set of per-chain draws. Based on paper
 * https://arxiv.org/abs/1903.08008   Computes bulk ESS over entire sample,
 * and tail ESS over the 0.05 and 0.95 quantiles.
 *
 * When the number of total draws N is odd, the last draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation

 * @param chains matrix of per-chain draws, num_iters X chain
 * @return pair ESS_bulk, ESS_tail
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
