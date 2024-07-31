#ifndef STAN_ANALYZE_MCMC_CHECK_CHAINS_HPP
#define STAN_ANALYZE_MCMC_CHECK_CHAINS_HPP

#include <stan/math/prim.hpp>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace stan {
namespace analyze {

/**
 * Checks that values across all matrix columns finite and non-identical.
 *
 * @param chains matrix of draws, one column per chain
 * @return bool true if OK, false otherwise
 */
inline bool is_finite_and_varies(const Eigen::MatrixXd chains) {
  size_t num_chains = chains.cols();
  size_t num_samples = chains.rows();
  Eigen::VectorXd first_draws = Eigen::VectorXd::Zero(num_chains);
  for (std::size_t i = 0; i < num_chains; ++i) {
    first_draws(i) = chains.col(i)(0);
    for (int j = 0; j < num_samples; ++j) {
      if (!std::isfinite(chains.col(i)(j)))
        return false;
    }
    if (chains.col(i).isApproxToConstant(first_draws(i))) {
      return false;
    }
  }
  if (num_chains > 1 && first_draws.isApproxToConstant(first_draws(0))) {
    return false;
  }
  return true;
}

}  // namespace analyze
}  // namespace stan

#endif
