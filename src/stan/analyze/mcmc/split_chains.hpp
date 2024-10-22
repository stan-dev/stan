#ifndef STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP
#define STAN_ANALYZE_MCMC_SPLIT_CHAINS_HPP

#include <stan/math/prim.hpp>
#include <cmath>
#include <vector>
#include <algorithm>

namespace stan {
namespace analyze {

/**
 * Splits each chain into two chains of equal length.  When the
 * number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * @param chains vector of per-chain sample matrices
 * @param index matrix column for parameter of interest
 * @return samples matrix, shape  (num_iters/2, num_chains*2)
 */
inline Eigen::MatrixXd split_chains(const std::vector<Eigen::MatrixXd>& chains,
                                    const int index) {
  size_t num_chains = chains.size();
  size_t num_draws = chains[0].rows();
  size_t half = std::floor(num_draws / 2.0);
  size_t tail_start = std::floor((num_draws + 1) / 2.0);

  Eigen::MatrixXd split_draws_matrix(half, num_chains * 2);
  int split_i = 0;
  for (std::size_t i = 0; i < num_chains; ++i) {
    Eigen::Map<const Eigen::VectorXd> head_block(chains[i].col(index).data(),
                                                 half);
    Eigen::Map<const Eigen::VectorXd> tail_block(
        chains[i].col(index).data() + tail_start, half);

    split_draws_matrix.col(split_i) = head_block;
    split_draws_matrix.col(split_i + 1) = tail_block;
    split_i += 2;
  }
  return split_draws_matrix;
}

/**
 * Splits each chain into two chains of equal length.  When the
 * number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * @param samples matrix of per-chain samples, shape (num_iters, num_chains)
 * @return samples matrix reshaped as (num_iters/2, num_chains*2)
 */
inline Eigen::MatrixXd split_chains(const Eigen::MatrixXd& samples) {
  size_t num_chains = samples.cols();
  size_t num_draws = samples.rows();
  size_t half = std::floor(num_draws / 2.0);
  size_t tail_start = std::floor((num_draws + 1) / 2.0);

  Eigen::MatrixXd split_draws_matrix(half, num_chains * 2);
  int split_i = 0;
  for (std::size_t i = 0; i < num_chains; ++i) {
    Eigen::Map<const Eigen::VectorXd> head_block(samples.col(i).data(), half);
    Eigen::Map<const Eigen::VectorXd> tail_block(
        samples.col(i).data() + tail_start, half);

    split_draws_matrix.col(split_i) = head_block;
    split_draws_matrix.col(split_i + 1) = tail_block;
    split_i += 2;
  }
  return split_draws_matrix;
}

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
