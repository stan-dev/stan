#ifndef STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <cmath>
#include <vector>
#include <utility>

namespace stan {
namespace analyze {

  inline
  Eigen::MatrixXd
  split_chains(const std::vector<const double*>& draws,
               const std::vector<size_t>& sizes) {

    int num_chains = sizes.size();

    // need to generalize to each jagged draws per chain
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    double half = num_draws / 2.0;
    int half_f = std::floor(half);
    int half_c = std::ceil(half);
    Eigen::MatrixXd split_draws(half_f, 2 * num_chains);
    for (int n = 0; n < num_chains; ++n) {
       split_draws.col(2*n) = Eigen::VectorXd::Map(&draws[n][0], half_f);
       split_draws.col(2*n + 1) = Eigen::VectorXd::Map(&draws[n][half_c], half_f);
  }

    return split_draws;
  }

} // namespace analyze
} // namespace stan

#endif
