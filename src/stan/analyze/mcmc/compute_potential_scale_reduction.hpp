#ifndef STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP
#define STAN_ANALYZE_MCMC_COMPUTE_POTENTIAL_SCALE_REDUCTION_HPP

#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/autocovariance.hpp>
#include <stan/analyze/mcmc/split_chains.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/math/distributions/normal.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes normalized average ranks for draws. Transforming them to normal
 * scores using inverse normal transformation and a fractional offset. Based on
 * paper https://arxiv.org/abs/1903.08008
 * @param chains stores chains in columns
 * @return normal scores for average ranks of draws
 */
inline Eigen::MatrixXd rank_transform(const Eigen::MatrixXd& chains) {
  const Eigen::Index rows = chains.rows();
  const Eigen::Index cols = chains.cols();
  const Eigen::Index size = rows * cols;

  std::vector<std::pair<double, int>> value_with_index(size);

  for (Eigen::Index i = 0; i < size; ++i) {
    value_with_index[i] = {chains(i), i};
  }

  std::sort(value_with_index.begin(), value_with_index.end());

  Eigen::MatrixXd rank_matrix = Eigen::MatrixXd::Zero(rows, cols);

  // Assigning average ranks
  for (Eigen::Index i = 0; i < size; ++i) {
    // Handle ties by averaging ranks
    Eigen::Index j = i + 1;
    double sum_ranks = j;
    Eigen::Index count = 1;

    while (j < size && value_with_index[j].first == value_with_index[i].first) {
      sum_ranks += j + 1;  // Rank starts from 1
      ++j;
      ++count;
    }
    double avg_rank = sum_ranks / count;
    boost::math::normal_distribution<double> dist;
    for (std::size_t k = i; k < j; ++k) {
      double p = (avg_rank - 3.0 / 8.0) / (size - 2.0 * 3.0 / 8.0 + 1.0);
      const Eigen::Index index = value_with_index[k].second;
      rank_matrix(index) = boost::math::quantile(dist, p);
    }
    i = j - 1;  // Skip over tied elements
  }
  return rank_matrix;
}

/**
 * Computes square root of marginal posterior variance of the estimand by the
 * weigted average of within-chain variance W and between-chain variance B.
 *
 * @param chains stores chains in columns
 * @return square root of ((N-1)/N)W + B/N
 */
inline double rhat(const Eigen::MatrixXd& chains) {
  const Eigen::Index num_chains = chains.cols();
  const Eigen::Index num_draws = chains.rows();

  Eigen::RowVectorXd within_chain_means = chains.colwise().mean();
  double across_chain_mean = within_chain_means.mean();
  double between_variance
      = num_draws
        * (within_chain_means.array() - across_chain_mean).square().sum()
        / (num_chains - 1);
  double within_variance =
      // Divide each row by chains and get sum of squares for each chain
      // (getting a vector back)
      ((chains.rowwise() - within_chain_means)
           .array()
           .square()
           .colwise()
           // divide each sum of square by num_draws, sum the sum of squares,
           // and divide by num chains
           .sum()
       / (num_draws - 1.0))
          .sum()
      / num_chains;

  return sqrt((between_variance / within_variance + num_draws - 1) / num_draws);
}

/**
 * Computes the potential scale reduction (Rhat) using rank based diagnostic for
 * the specified parameter across all kept samples. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param chain_begins stores pointers to arrays of chains
 * @param chain_sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline std::pair<double, double> compute_potential_scale_reduction_rank(
    const std::vector<const double*>& chain_begins,
    const std::vector<size_t>& chain_sizes) {
  std::vector<const double*> nonzero_chain_begins;
  std::vector<std::size_t> nonzero_chain_sizes;
  nonzero_chain_begins.reserve(chain_begins.size());
  nonzero_chain_sizes.reserve(chain_sizes.size());
  for (size_t i = 0; i < chain_sizes.size(); ++i) {
    if (chain_sizes[i]) {
      nonzero_chain_begins.push_back(chain_begins[i]);
      nonzero_chain_sizes.push_back(chain_sizes[i]);
    }
  }
  if (!nonzero_chain_sizes.size()) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }
  std::size_t num_nonzero_chains = nonzero_chain_sizes.size();
  std::size_t min_num_draws = nonzero_chain_sizes[0];
  for (std::size_t chain = 1; chain < num_nonzero_chains; ++chain) {
    min_num_draws = std::min(min_num_draws, nonzero_chain_sizes[chain]);
  }

  // check if chains are constant; all equal to first draw's value
  bool are_all_const = false;
  Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_nonzero_chains);
  Eigen::MatrixXd draws_matrix(min_num_draws, num_nonzero_chains);

  for (std::size_t chain = 0; chain < num_nonzero_chains; chain++) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draws(
        nonzero_chain_begins[chain], nonzero_chain_sizes[chain]);

    for (std::size_t n = 0; n < min_num_draws; n++) {
      if (!std::isfinite(draws(n))) {
        return {std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
      }
      draws_matrix(n, chain) = draws(n);
    }

    init_draw(chain) = draws(0);
    are_all_const |= !draws.isApproxToConstant(draws(0));
  }
  // If all chains are constant then return NaN
  if (are_all_const && init_draw.isApproxToConstant(init_draw(0))) {
    return {std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN()};
  }

  double rhat_bulk = rhat(rank_transform(draws_matrix));
  double rhat_tail = rhat(rank_transform(
      (draws_matrix.array() - math::quantile(draws_matrix.reshaped(), 0.5))
          .abs()));

  return std::make_pair(rhat_bulk, rhat_tail);
}

/**
 * Computes the potential scale reduction (Rhat) for the specified
 * parameter across all kept samples.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_potential_scale_reduction(
    std::vector<const double*> draws, std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  if (num_draws == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  // check if chains are constant; all equal to first draw's value
  bool are_all_const = false;
  Eigen::VectorXd init_draw = Eigen::VectorXd::Zero(num_chains);

  for (int chain = 0; chain < num_chains; chain++) {
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1>> draw(
        draws[chain], sizes[chain]);

    for (int n = 0; n < num_draws; n++) {
      if (!std::isfinite(draw(n))) {
        return std::numeric_limits<double>::quiet_NaN();
      }
    }

    init_draw(chain) = draw(0);

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

  using boost::accumulators::accumulator_set;
  using boost::accumulators::stats;
  using boost::accumulators::tag::mean;
  using boost::accumulators::tag::variance;

  Eigen::VectorXd chain_mean(num_chains);
  accumulator_set<double, stats<variance>> acc_chain_mean;
  Eigen::VectorXd chain_var(num_chains);
  double unbiased_var_scale = num_draws / (num_draws - 1.0);

  for (int chain = 0; chain < num_chains; ++chain) {
    accumulator_set<double, stats<mean, variance>> acc_draw;
    for (int n = 0; n < num_draws; ++n) {
      acc_draw(draws[chain][n]);
    }

    chain_mean(chain) = boost::accumulators::mean(acc_draw);
    acc_chain_mean(chain_mean(chain));
    chain_var(chain)
        = boost::accumulators::variance(acc_draw) * unbiased_var_scale;
  }

  double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
                       * num_chains / (num_chains - 1);
  double var_within = chain_var.mean();

  // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
  return sqrt((var_between / var_within + num_draws - 1) / num_draws);
}

/**
 * Computes the potential scale reduction (Rhat) using rank based diagnostic for
 * the specified parameter across all kept samples. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory. Chains are trimmed from the back to match the
 * length of the shortest chain. Argument size will be broadcast to
 * same length as draws.
 *
 * @param chain_begins stores pointers to arrays of chains
 * @param size stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline std::pair<double, double> compute_potential_scale_reduction_rank(
    const std::vector<const double*>& chain_begins, size_t size) {
  std::vector<size_t> sizes(chain_begins.size(), size);
  return compute_potential_scale_reduction_rank(chain_begins, sizes);
}

/**
 * Computes the potential scale reduction (Rhat) for the specified
 * parameter across all kept samples.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Argument size will be broadcast to
 * same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param size stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_potential_scale_reduction(
    std::vector<const double*> draws, size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_potential_scale_reduction(draws, sizes);
}

/**
 * Computes the potential scale reduction (Rhat) using rank based diagnostic for
 * the specified parameter across all kept samples. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * When the number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory. Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param chain_begins stores pointers to arrays of chains
 * @param chain_sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline std::pair<double, double> compute_split_potential_scale_reduction_rank(
    const std::vector<const double*>& chain_begins,
    const std::vector<size_t>& chain_sizes) {
  size_t num_chains = chain_sizes.size();
  size_t num_draws = chain_sizes[0];
  for (size_t chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, chain_sizes[chain]);
  }

  std::vector<const double*> split_draws
      = split_chains(chain_begins, chain_sizes);

  size_t half = std::floor(num_draws / 2.0);
  std::vector<size_t> half_sizes(2 * num_chains, half);

  return compute_potential_scale_reduction_rank(split_draws, half_sizes);
}

/**
 * Computes the split potential scale reduction (Rhat) for the
 * specified parameter across all kept samples.  When the number of
 * total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.
 *
 * @param draws stores pointers to arrays of chains
 * @param sizes stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_split_potential_scale_reduction(
    std::vector<const double*> draws, std::vector<size_t> sizes) {
  int num_chains = sizes.size();
  size_t num_draws = sizes[0];
  for (int chain = 1; chain < num_chains; ++chain) {
    num_draws = std::min(num_draws, sizes[chain]);
  }

  std::vector<const double*> split_draws = split_chains(draws, sizes);

  double half = num_draws / 2.0;
  std::vector<size_t> half_sizes(2 * num_chains, std::floor(half));

  return compute_potential_scale_reduction(split_draws, half_sizes);
}

/**
 * Computes the potential scale reduction (Rhat) using rank based diagnostic for
 * the specified parameter across all kept samples. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * When the number of total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Argument size will be broadcast to
 * same length as draws.
 *
 * @param chain_begins stores pointers to arrays of chains
 * @param size stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline std::pair<double, double> compute_split_potential_scale_reduction_rank(
    const std::vector<const double*>& chain_begins, size_t size) {
  size_t num_chains = chain_begins.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_split_potential_scale_reduction_rank(chain_begins, sizes);
}

/**
 * Computes the split potential scale reduction (Rhat) for the
 * specified parameter across all kept samples.  When the number of
 * total draws N is odd, the (N+1)/2th draw is ignored.
 *
 * See more details in Stan reference manual section "Potential
 * Scale Reduction". http://mc-stan.org/users/documentation
 *
 * Current implementation assumes draws are stored in contiguous
 * blocks of memory.  Chains are trimmed from the back to match the
 * length of the shortest chain.  Argument size will be broadcast to
 * same length as draws.
 *
 * @param draws stores pointers to arrays of chains
 * @param size stores sizes of chains
 * @return potential scale reduction for the specified parameter
 */
inline double compute_split_potential_scale_reduction(
    std::vector<const double*> draws, size_t size) {
  int num_chains = draws.size();
  std::vector<size_t> sizes(num_chains, size);
  return compute_split_potential_scale_reduction(draws, sizes);
}

}  // namespace analyze
}  // namespace stan

#endif
