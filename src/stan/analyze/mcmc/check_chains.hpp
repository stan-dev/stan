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
 * Checks that all values are finite and that at least one chain is not
 * constant.  Some chains can be constant: a chain that does not move gives
 * a large R-hat and a small ESS, which the diagnostics must report.
 *
 * @param chains matrix of draws, one column per chain
 * @return bool true if OK, false otherwise
 */
inline bool is_finite_and_varies(const Eigen::MatrixXd& chains) {
  if (chains.rows() < 2 || !chains.allFinite()) {
    return false;
  }
  for (Eigen::Index i = 0; i < chains.cols(); ++i) {
    if (!chains.col(i).isApproxToConstant(chains(0, i))) {
      return true;
    }
  }
  return false;
}

}  // namespace analyze
}  // namespace stan

#endif
