#ifndef STAN_MCMC_CHAINSET_HPP
#define STAN_MCMC_CHAINSET_HPP

#include <stan/io/stan_csv_reader.hpp>
#include <stan/math/prim.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/tail_quantile.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/covariance.hpp>
#include <boost/accumulators/statistics/variates/covariate.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>
#include <cstdlib>

namespace stan {
namespace mcmc {
using Eigen::Dynamic;

/**
 * An <code>mcmc::chainset</code> object manages the post-warmup draws
 * across a set of MCMC chains, which all have the same number or samples.
 *
 * <p><b>Storage Order</b>: Storage is column/last-index major.
 */
class chainset {
 private:
  size_t num_samples_;
  std::vector<std::string> param_names_;
  std::vector<Eigen::MatrixXd> chains_;

  static size_t thinned_samples(const stan::io::stan_csv& stan_csv) {
    size_t thinned_samples = stan_csv.metadata.num_samples;
    if (stan_csv.metadata.thin > 1) {
      thinned_samples = thinned_samples / stan_csv.metadata.thin;
    }
    return thinned_samples;
  }

  static bool is_valid(const stan::io::stan_csv& stan_csv) {
    if (stan_csv.header.empty()) {
      return false;
    }
    if (stan_csv.samples.size() == 0) {
      return false;
    }
    if (stan_csv.samples.rows() != thinned_samples(stan_csv)) {
      return false;
    }
    return true;
  }

  /**
   * Process first chain: record header, thinned samples,
   * add samples to vector chains.
   */
  void init_from_stan_csv(const stan::io::stan_csv& stan_csv) {
    if (!is_valid(stan_csv)) {
      throw std::invalid_argument("Invalid sample");
    }
    if (chains_.size() > 0) {
      throw std::invalid_argument("Cannot re-initialize chains object");
    }
    param_names_ = stan_csv.header;
    num_samples_ = thinned_samples(stan_csv);
    chains_.push_back(stan_csv.samples);
  }

  /**
   * Process next chain: validate size, shape, column names,
   * append to vector chains.
   */
  void add(const stan::io::stan_csv& stan_csv) {
    if (!is_valid(stan_csv)) {
      throw std::invalid_argument("Invalid sample");
    }
    if (stan_csv.header.size() != num_params()) {
      throw std::invalid_argument(
          "Error add(stan_csv): number of columns in"
          " sample does not match first chain");
    }
    if (thinned_samples(stan_csv) != num_samples_) {
      throw std::invalid_argument(
          "Error add(stan_csv): number of sampling draws in"
          " sample does not match first chain");
    }
    for (int i = 0; i < num_params(); i++) {
      if (param_names_[i] != stan_csv.header[i]) {
        std::stringstream ss;
        ss << "Error add(stan_csv): header " << param_names_[i]
           << " does not match chain's header (" << stan_csv.header[i] << ")";
        throw std::invalid_argument(ss.str());
      }
    }
    chains_.push_back(stan_csv.samples);
  }

 public:
  explicit chainset(const stan::io::stan_csv& stan_csv) {
    init_from_stan_csv(stan_csv);
  }

  explicit chainset(const std::vector<stan::io::stan_csv>& stan_csv) {
    if (stan_csv.empty())
      return;
    init_from_stan_csv(stan_csv[0]);
    for (size_t i = 1; i < stan_csv.size(); ++i) {
      add(stan_csv[i]);
    }
  }

  /**
   * Report number of chains in chainset.
   * @return chainset size.
   */
  inline int num_chains() const { return chains_.size(); }

  /**
   * Report number of parameters per chain.
   * @return size of parameter names vector.
   */
  inline int num_params() const { return param_names_.size(); }

  /**
   * Report number of samples (draws) per chain.
   * @return rows per chain
   */
  inline int num_samples() const { return num_samples_; }

  /**
   * Get parameter names.
   * @return vector of parameter names.
   */
  const std::vector<std::string>& param_names() const { return param_names_; }

  /**
   * Get name of parameter at specified column index.
   * Throws exception if index is out of bounds.
   *
   * @param index column index
   * @return parameter name
   */
  const std::string& param_name(int index) const {
    if (index < 0 || index >= param_names_.size()) {
      std::stringstream ss;
      ss << "Bad index " << index << ", should be between 0 and "
         << (param_names_.size() - 1);
      throw std::invalid_argument(ss.str());
    }
    return param_names_[index];
  }

  /**
   * Get column index for specified parameter name.
   * Throws exception if name not found.
   *
   * @param name parameter name
   * @return column index
   */
  int index(const std::string& name) const {
    auto it = std::find(param_names_.begin(), param_names_.end(), name);
    if (it == param_names_.end()) {
      std::stringstream ss;
      ss << "Unknown parameter name " << name;
      throw std::invalid_argument(ss.str());
    }
    return std::distance(param_names_.begin(), it);
  }

  /**
   * Assemble samples (draws) from specified column index across all chains
   * into a matrix of samples X chain.
   * Throws exception if column index is out of bounds.
   *
   * @param index column index
   * @return matrix of draws across all chains
   */
  Eigen::MatrixXd samples(const int index) const {
    Eigen::MatrixXd result(num_samples(), chains_.size());
    if (index < 0 || index >= param_names_.size()) {
      std::stringstream ss;
      ss << "Bad index " << index << ", should be between 0 and "
         << (param_names_.size() - 1);
      throw std::invalid_argument(ss.str());
    }
    for (int i = 0; i < chains_.size(); ++i) {
      result.col(i) = chains_[i].col(index);
    }
    return result;
  }

  /**
   * Assemble samples (draws) from specified parameter name across all chains
   * into a matrix of samples X chain.
   * Throws exception if parameter name is not found.
   *
   * @param name parameter name
   * @return matrix of draws across all chains
   */
  Eigen::MatrixXd samples(const std::string& name) const {
    return samples(index(name));
  }

  /**
   * Compute mean value for specified parameter across all chains.
   *
   * @param index parameter index
   * @return mean parameter value
   */
  double mean(const int index) const { return samples(index).mean(); }

  /**
   * Compute mean value for specified parameter across all chains.
   *
   * @param name parameter name
   * @return mean parameter value
   */
  double mean(const std::string& name) const { return mean(index(name)); }

  /**
   * Compute sample variance for specified parameter across all chains.
   * 1 / (N - 1) * sum((theta_n - mean(theta))^2)
   *
   * @param index parameter index
   * @return sample variance
   */
  double variance(const int index) const {
    Eigen::MatrixXd draws = samples(index);
    return (draws.array() - draws.mean()).square().sum() / (draws.size() - 1);
  }

  /**
   * Compute sample variance for specified parameter across all chains.
   * 1 / (N - 1) * sum((theta_n - mean(theta))^2)
   *
   * @param name parameter name
   * @return sample variance
   */
  double variance(const std::string& name) const {
    return variance(index(name));
  }

  /**
   * Compute standard deviation for specified parameter across all chains.
   *
   * @param index parameter index
   * @return sample sd
   */
  double sd(const int index) const { return std::sqrt(variance(index)); }

  /**
   * Compute standard deviation for specified parameter across all chains.
   *
   * @param name parameter name
   * @return sample sd
   */
  double sd(const std::string& name) const { return sd(index(name)); }

  /**
   * Compute median value of specified parameter across all chains.
   *
   * @param index parameter index
   * @return median
   */
  double median(const int index) const {
    Eigen::MatrixXd draws = samples(index);
    std::vector<double> sorted(draws.data(), draws.data() + draws.size());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(0.5 * (sorted.size() - 1));
    return sorted[idx];
  }

  /**
   * Compute median value of specified parameter across all chains.
   *
   * @param name parameter name
   * @return median
   */
  double median(const std::string& name) const { return median(index(name)); }

  /**
   * Compute maximum absolute deviation (mad) for specified parameter.
   *
   * Follows R implementation:  constant * median(abs(x - center))
   * where the value of center is median(x) and the constant is 1.4826,
   * a scale factor for asymptotically normal consistency: `1/qnorm(3/4)`.
   * (R stats version 3.6.2)
   *
   * @param index parameter index
   * @return sample mad
   */
  double max_abs_deviation(const int index) const {
    Eigen::MatrixXd draws = samples(index);
    auto center = median(index);
    Eigen::MatrixXd abs_dev = (draws.array() - center).abs();
    std::vector<double> sorted(abs_dev.data(), abs_dev.data() + abs_dev.size());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(0.5 * (sorted.size() - 1));
    return 1.4826 * sorted[idx];
  }

  /**
   * Compute maximum absolute deviation (mad) for specified parameter.
   *
   * Follows R implementation:  constant * median(abs(x - center))
   * where the value of center is median(x) and the constant is 1.4826,
   * a scale factor for asymptotically normal consistency: `1/qnorm(3/4)`.
   * (R stats version 3.6.2)
   *
   * @param name parameter name
   * @return sample mad
   */
  double max_abs_deviation(const std::string& name) const {
    return max_abs_deviation(index(name));
  }

  /**
   * Compute the quantile value of the specified parameter
   * at the specified probability.
   *
   * Throws exception if specified probability is not between 0 and 1.
   *
   * @param index parameter index
   * @param prob probability
   * @return parameter value at quantile
   */
  double quantile(const int index, const double prob) const {
    // Ensure the probability is within [0, 1]
    if (prob < 0.0 || prob > 1.0) {
      throw std::out_of_range("Probability must be between 0 and 1.");
    }
    Eigen::MatrixXd draws = samples(index);
    std::vector<double> sorted(draws.data(), draws.data() + draws.size());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(prob * (sorted.size() - 1));
    return sorted[idx];
  }

  /**
   * Compute the quantile value of the specified parameter
   * at the specified probability.
   *
   * Throws exception if specified probability is not between 0 and 1.
   *
   * @param name parameter name
   * @param prob probability
   * @return parameter value at quantile
   */
  double quantile(const std::string& name, const double prob) const {
    return quantile(index(name), prob);
  }

  /**
   * Compute the quantile values of the specified parameter
   * for a set of specified probabilities.
   *
   * Throws exception if any probability is not between 0 and 1.
   *
   * @param index parameter index
   * @param probs vector of probabilities
   * @return vector of parameter values for quantiles
   */
  Eigen::VectorXd quantiles(const int index,
                            const Eigen::VectorXd& probs) const {
    Eigen::VectorXd quantiles(probs.size());
    if (probs.size() == 0)
      return quantiles;
    if (probs.minCoeff() < 0.0 || probs.maxCoeff() > 1.0) {
      throw std::out_of_range("Probabilities must be between 0 and 1.");
    }
    Eigen::MatrixXd draws = samples(index);
    std::vector<double> sorted(draws.data(), draws.data() + draws.size());
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < probs.size(); ++i) {
      size_t idx = static_cast<size_t>(probs[i] * (sorted.size() - 1));
      quantiles[i] = sorted[idx];
    }
    return quantiles;
  }

  /**
   * Compute the quantile values of the specified parameter
   * for a set of specified probabilities.
   *
   * Throws exception if any probability is not between 0 and 1.
   *
   * @param name parameter name
   * @param probs vector of probabilities
   * @return vector of parameter values for quantiles
   */
  Eigen::VectorXd quantiles(const std::string& name,
                            const Eigen::VectorXd& probs) const {
    return quantiles(index(name), probs);
  }

  /**
   * Computes the split potential scale reduction (split Rhat) using rank based
   * diagnostic for a set of per-chain draws, for bulk and tail Rhat.
   * Based on paper https://arxiv.org/abs/1903.08008
   *
   * @param index parameter index
   * @return pair (bulk_rhat, tail_rhat)
   */
  std::pair<double, double> split_rank_normalized_rhat(const int index) const {
    return analyze::split_rank_normalized_rhat(samples(index));
  }

  /**
   * Computes the split potential scale reduction (split Rhat) using rank based
   * diagnostic for a set of per-chain draws, for bulk and tail Rhat.
   * Based on paper https://arxiv.org/abs/1903.08008
   *
   * @param name parameter name
   * @return pair (bulk_rhat, tail_rhat)
   */
  std::pair<double, double> split_rank_normalized_rhat(
      const std::string& name) const {
    return split_rank_normalized_rhat(index(name));
  }

  /**
   * Computes the effective sample size (ESS) for the specified
   * parameter across all chains, according to the algorithm presented in
   * https://arxiv.org/abs/1903.08008, section 3.2 for folded (split)
   * rank-normalized ESS for both bulk and tail ESS.
   *
   * @param index parameter index
   * @return pair (bulk_ess, tail_ess)
   */
  std::pair<double, double> split_rank_normalized_ess(const int index) const {
    return analyze::split_rank_normalized_ess(samples(index));
  }

  /**
   * Computes the effective sample size (ESS) for the specified
   * parameter across all chains, according to the algorithm presented in
   * https://arxiv.org/abs/1903.08008, section 3.2 for folded (split)
   * rank-normalized ESS for both bulk and tail ESS.
   *
   * @param name parameter name
   * @return pair (bulk_ess, tail_ess)
   */
  std::pair<double, double> split_rank_normalized_ess(
      const std::string& name) const {
    return split_rank_normalized_ess(index(name));
  }

  /**
   * Computes the mean Monte Carlo error estimate for the central 90% interval.
   * See https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package
   *
   * @param index parameter index
   * @return pair (bulk_ess, tail_ess)
   */
  double mcse_mean(const int index) const {
    double ess_bulk = analyze::split_rank_normalized_ess(samples(index)).first;
    return sd(index) / std::sqrt(ess_bulk);
  }

  /**
   * Computes the mean Monte Carlo error estimate for the central 90% interval.
   * See https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package.
   *
   * @param name parameter name
   * @return pair (bulk_ess, tail_ess)
   */
  double mcse_mean(const std::string& name) const {
    return mcse_mean(index(name));
  }

  /**
   * Computes the standard deviation of the Monte Carlo error estimate
   * https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package.
   *
   * @param index parameter index
   * @return pair (bulk_ess, tail_ess)
   */
  double mcse_sd(const int index) const {
    Eigen::MatrixXd s = samples(index);
    Eigen::MatrixXd s2 = s.array().square();
    double ess_s = analyze::split_rank_normalized_ess(s).first;
    double ess_s2 = analyze::split_rank_normalized_ess(s2).first;
    double ess_sd = std::min(ess_s, ess_s2);
    return sd(index)
           * std::sqrt(std::exp(1) * std::pow(1 - 1 / ess_sd, ess_sd - 1) - 1);
  }

  /**
   * Computes the standard deviation of the Monte Carlo error estimate
   * https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package
   *
   * @param name parameter name
   * @return pair (bulk_ess, tail_ess)
   */
  double mcse_sd(const std::string& name) const { return mcse_sd(index(name)); }

  /**
   * Compute autocorrelation for one column of one chain.
   * Throws exception if column index is out of bounds.
   * Autocorrelation is computed using Stan math library implmentation.
   *
   * @param chain chain index
   * @param index column index
   * @return vector of chain autocorrelation at all lags
   */
  Eigen::VectorXd autocorrelation(const int chain, const int index) const {
    if (chain < 0 || chain >= num_chains()) {
      std::stringstream ss;
      ss << "Bad index " << index << ", should be between 0 and "
         << (num_chains() - 1);
      throw std::invalid_argument(ss.str());
    }
    if (index < 0 || index >= param_names().size()) {
      std::stringstream ss;
      ss << "Bad index " << index << ", should be between 0 and "
         << (num_params() - 1);
      throw std::invalid_argument(ss.str());
    }
    Eigen::MatrixXd s = samples(index);
    Eigen::Map<const Eigen::VectorXd> chain_col(samples(chain).data(),
                                                num_samples());
    Eigen::VectorXd autocorr_col(num_samples());
    stan::math::autocorrelation<double>(s.col(chain), autocorr_col);
    return autocorr_col;
  }

  /**
   * Compute autocorrelation for one column of one chain.
   * Throws exception if column index is out of bounds.
   * Autocorrelation is computed using Stan math library implmentation.
   *
   * @param chain chain index
   * @param name column name
   * @return vector of chain autocorrelation at all lags
   */
  Eigen::VectorXd autocorrelation(const int chain,
                                  const std::string name) const {
    return autocorrelation(chain, index(name));
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
