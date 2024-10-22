#ifndef STAN_ANALYZE_MCMC_RANK_NORMALIZATION_HPP
#define STAN_ANALYZE_MCMC_RANK_NORMALIZATION_HPP

#include <stan/math/prim.hpp>
#include <boost/math/distributions/normal.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits>

namespace stan {
namespace analyze {

/**
 * Computes normalized average ranks for pooled draws.  The values across
 * all draws be finite and not constant. Normal scores computed using
 * inverse normal transformation and a fractional offset. Based on paper
 * https://arxiv.org/abs/1903.08008
 *
 * @param chains matrix of draws, one column per chain
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
      double p = (avg_rank - 0.375) / (size + 0.25);
      const Eigen::Index index = value_with_index[k].second;
      rank_matrix(index) = boost::math::quantile(dist, p);
    }
    i = j - 1;  // Skip over tied elements
  }
  return rank_matrix;
}

}  // namespace analyze
}  // namespace stan

#endif
