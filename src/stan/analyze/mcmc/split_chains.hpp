#ifndef STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP

#include <cmath>
#include <vector>
#include <algorithm>

namespace stan {
namespace analyze {

/**
 * Splits each chain into two chains of equal length.  When the
 * number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes chains are all of equal size and
 * draws are stored in contiguous blocks of memory.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return std::vector of pointers to twice as many arrays of half chains
 */
inline std::vector<const double*> split_chains(
    const std::vector<const double*>& draws, const std::vector<size_t>& sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  double half = num_draws / 2.0;
  int half_draws = std::ceil(half);
  std::vector<const double*> split_draws(2 * num_chains);
  for (int n = 0; n < num_chains; ++n) {
    split_draws[2 * n] = &draws[n][0];
    split_draws[2 * n + 1] = &draws[n][half_draws];
  }

  return split_draws;
}

/**
 * Splits each chain into two chains of equal length.  When the
 * number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Effective
 * Sample Size". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes chains are all of equal size and
 * draws are stored in contiguous blocks of memory.  Argument size
 * will be broadcast to same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param size size of chains
 * @return std::vector of pointers to twice as many arrays of half chains
 */
inline std::vector<const double*> split_chains(std::vector<const double*> draws,
                                               size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return split_chains(draws, sizes);
}

}  // namespace analyze
}  // namespace stan

#endif
