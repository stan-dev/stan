#ifndef STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP

#include <cmath>
#include <vector>

namespace stan {
namespace analyze {

  inline
  std::vector<const double*>
  split_chains(const std::vector<const double*>& draws,
               const std::vector<size_t>& sizes) {

    int num_chains = sizes.size();

    // need to generalize to each jagged draws per chain
    size_t num_draws = sizes[0];
    for (int chain = 1; chain < num_chains; ++chain) {
      num_draws = std::min(num_draws, sizes[chain]);
    }

    double half = num_draws / 2.0;
    int half_draws = std::ceil(half);
    std::vector<const double*> split_draws(2 * num_chains);
    for (int n = 0; n < num_chains; ++n) {
       split_draws[2*n] = &draws[n][0];
       split_draws[2*n + 1] = &draws[n][half_draws];
  }

    return split_draws;
  }

} // namespace analyze
} // namespace stan

#endif
