#ifndef STAN_MCMC_CHAINSET_HPP
#define STAN_MCMC_CHAINSET_HPP

#include <stan/io/stan_csv_reader.hpp>
#include <stan/math/prim.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/quantile.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_ess.hpp>
#include <stan/analyze/mcmc/split_rank_normalized_rhat.hpp>
#include <stan/analyze/mcmc/mcse.hpp>
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

/**
 * A <code>mcmc::chainset</code> object manages the post-warmup draws
 * across a set of MCMC chains, which all have the same number of samples.
 *
 * @note samples are stored in column major, i.e., each column corresponds to
 * an output variable (element).
 *
 */
class chainset {
 private:
  size_t num_samples_;
  std::vector<std::string> param_names_;
  std::vector<Eigen::MatrixXd> chains_;

 public:
  /* Construct a chainset from a single sample.
   * Throws execption if sample is empty.
   */
  explicit chainset(const stan::io::stan_csv& stan_csv) {
    if (chains_.size() > 0) {
      throw std::invalid_argument("Cannot re-initialize chains object");
    }
    if (stan_csv.header.size() == 0 || stan_csv.samples.rows() == 0) {
      throw std::invalid_argument("Error: empty sample");
    }
    param_names_ = stan_csv.header;
    num_samples_ = stan_csv.samples.rows();
    chains_.push_back(stan_csv.samples);
  }

  /* Construct a chainset from a set of samples.
   * Throws execption if sample column names and shapes don't match.
   */
  explicit chainset(const std::vector<stan::io::stan_csv>& stan_csv) {
    if (stan_csv.empty())
      return;
    if (chains_.size() > 0) {
      throw std::invalid_argument("Cannot re-initialize chains object");
    }
    if (stan_csv[0].header.size() == 0 || stan_csv[0].samples.rows() == 0) {
      throw std::invalid_argument("Error: empty sample");
    }
    param_names_ = stan_csv[0].header;
    num_samples_ = stan_csv[0].samples.rows();
    chains_.push_back(stan_csv[0].samples);
    std::stringstream ss;
    for (size_t i = 1; i < stan_csv.size(); ++i) {
      if (stan_csv[i].header.size() != param_names_.size()) {
        ss << "Error: chain " << (i + 1) << " missing or extra columns";
        throw std::invalid_argument(ss.str());
      }
      for (int j = 0; j < param_names_.size(); j++) {
        if (param_names_[j] != stan_csv[i].header[j]) {
          ss << "Error: chain " << (i + 1) << " header column " << (j + 1)
             << " doesn't match chain 1 header, found: "
             << stan_csv[i].header[j] << " expecting: " << param_names_[j];
          throw std::invalid_argument(ss.str());
        }
      }
      if (stan_csv[i].samples.rows() != num_samples_) {
        ss << "Error: chain " << (i + 1) << ", missing or extra rows.";
        throw std::invalid_argument(ss.str());
      }
      chains_.push_back(stan_csv[i].samples);
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
  double median(const int index) const { return (quantile(index, 0.5)); }

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
    Eigen::Map<Eigen::VectorXd> map(abs_dev.data(), abs_dev.size());
    return 1.4826 * stan::math::quantile(map, 0.5);
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
   * Calls stan::math::quantile which throws
   * std::invalid_argument If any element of samples_vec is NaN, or size 0.
   * and std::domain_error If `p<0` or `p>1`.
   *
   * @param index parameter index
   * @param prob probability
   * @return parameter value at quantile
   */
  double quantile(const int index, const double prob) const {
    // Ensure the probability is within [0, 1]
    Eigen::MatrixXd draws = samples(index);
    Eigen::Map<Eigen::VectorXd> map(draws.data(), draws.size());
    return stan::math::quantile(map, prob);
  }

  /**
   * Compute the quantile value of the specified parameter
   * at the specified probability.
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
   * @param index parameter index
   * @param probs vector of probabilities
   * @return vector of parameter values for quantiles
   */
  Eigen::VectorXd quantiles(const int index,
                            const Eigen::VectorXd& probs) const {
    if (probs.size() == 0)
      return Eigen::VectorXd::Zero(0);
    Eigen::MatrixXd draws = samples(index);
    Eigen::Map<Eigen::VectorXd> map(draws.data(), draws.size());
    std::vector<double> probs_vec(probs.data(), probs.data() + probs.size());
    std::vector<double> quantiles = stan::math::quantile(map, probs_vec);
    return Eigen::Map<Eigen::VectorXd>(quantiles.data(), quantiles.size());
  }

  /**
   * Compute the quantile values of the specified parameter
   * for a set of specified probabilities.
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
   * @return mcse
   */
  double mcse_mean(const int index) const {
    return analyze::mcse_mean(samples(index));
  }

  /**
   * Computes the mean Monte Carlo error estimate for the central 90% interval.
   * See https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package.
   *
   * @param name parameter name
   * @return mcse
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
   * @return mcse_sd
   */
  double mcse_sd(const int index) const {
    return analyze::mcse_sd(samples(index));
  }

  /**
   * Computes the standard deviation of the Monte Carlo error estimate
   * https://arxiv.org/abs/1903.08008, section 4.4.
   * Follows implementation in the R posterior package
   *
   * @param name parameter name
   * @return mcse_sd
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
