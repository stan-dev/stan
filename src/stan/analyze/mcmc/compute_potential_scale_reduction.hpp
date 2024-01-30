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

inline double median( Eigen::MatrixXd d){
    auto r { d.reshaped() };
    std::sort( r.begin(), r.end() );
    return r.size() % 2 == 0 ?
        r.segment( (r.size()-2)/2, 2 ).mean() :
        r( r.size()/2 );
}

Eigen::MatrixXd rankTransform(const Eigen::MatrixXd& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    int size = rows * cols;
    Eigen::MatrixXd rankMatrix = Eigen::MatrixXd::Zero(rows, cols);

    // Create a vector of pairs (value, original index)
    std::vector<std::pair<double, int>> valueWithIndex(size);

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            int index = col * rows + row; // Calculating linear index in column-major order
            valueWithIndex[index] = {matrix(row, col), index};
        }
    }

    // Sorting the pairs by value
    std::sort(valueWithIndex.begin(), valueWithIndex.end());

    // Assigning average ranks
    for (int i = 0; i < size; ++i) {
        // Handle ties by averaging ranks
        int j = i;
        double sumRanks = 0;
        int count = 0;

        while (j < size && valueWithIndex[j].first == valueWithIndex[i].first) {
            sumRanks += j + 1; // Rank starts from 1
            ++j;
            ++count;
        }

        double avgRank = sumRanks / count;
        for (int k = i; k < j; ++k) {
            int index = valueWithIndex[k].second;
            int row = index % rows; // Adjusting row index for column-major order
            int col = index / rows; // Adjusting column index for column-major order
            rankMatrix(row, col) = (avgRank - 3.0/8.0) / (size - 2.0 * 3.0/8.0 + 1.0);
            
        }
        i = j - 1; // Skip over tied elements
    }

    auto ndtri = [](double p) {
      boost::math::normal_distribution<double> dist; // Standard normal distribution
      return boost::math::quantile(dist, p); // Inverse CDF (quantile function)
    };

    rankMatrix = rankMatrix.unaryExpr(ndtri);
    return rankMatrix;
}

 
inline double rhat(const Eigen::MatrixXd& draws) {
  using boost::accumulators::accumulator_set;
  using boost::accumulators::stats;
  using boost::accumulators::tag::mean;
  using boost::accumulators::tag::variance;

  int num_chains = draws.cols();
  int num_draws = draws.rows();
  std::cout << num_chains << " " << num_draws << std::endl;
  Eigen::VectorXd chain_mean(num_chains);
  accumulator_set<double, stats<variance>> acc_chain_mean;
  Eigen::VectorXd chain_var(num_chains);
  double unbiased_var_scale = num_draws / (num_draws - 1.0);
  for (int chain = 0; chain < num_chains; ++chain) {
    accumulator_set<double, stats<mean, variance>> acc_draw;
    for (int n = 0; n < num_draws; ++n) {
      acc_draw(draws(n, chain));
    }
    chain_mean(chain) = boost::accumulators::mean(acc_draw);
    acc_chain_mean(chain_mean(chain));
    chain_var(chain) = boost::accumulators::variance(acc_draw) * unbiased_var_scale;
  }

  double var_between = num_draws * boost::accumulators::variance(acc_chain_mean)
                       * num_chains / (num_chains - 1);
  double var_within = chain_var.mean();

  // rewrote [(n-1)*W/n + B/n]/W as (n-1+ B/W)/n
  return sqrt((var_between / var_within + num_draws - 1) / num_draws);
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
    std::cout << "DRAWS POINTERS: " << std::endl;
    for (int i = 0; i < draws.size(); ++i) {
      std::cout << "Index: " << i << " P: " << draws[i] << " EXPECTED END: " << draws[i] + sizes[i] << std::endl;
    }
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

  Eigen::MatrixXd matrix(num_draws, num_chains);

    // Copy data from arrays to matrix
    for (int col = 0; col < num_chains; ++col) {
        for (int row = 0; row < num_draws; ++row) {
            matrix(row, col) = draws[col][row];
        }
    }

  double rhat_bulk = rhat(rankTransform(matrix));
  double rhat_tail = rhat(rankTransform((matrix.array() - median(matrix)).abs()));
  return std::max(rhat_bulk, rhat_tail);
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
