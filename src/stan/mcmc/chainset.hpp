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
template <typename Unused = void*>
class chainset {
 private:
  size_t num_samples_;
  std::vector<std::string> param_names_;
  std::vector<Eigen::MatrixXd> chains_;

  static size_t thinned_samples(const stan::io::stan_csv& stan_csv) {
    size_t thinned_samples = stan_csv.metadata.num_samples;
    if (stan_csv.metadata.thin > 0) {
      thinned_samples = thinned_samples / stan_csv.metadata.thin;
    }
    return thinned_samples;
  }

  static bool is_valid(const stan::io::stan_csv& stan_csv) {
    if (stan_csv.header.empty())
      return false;
    if (stan_csv.samples.size() == 0)
      return false;
    if (stan_csv.samples.rows() != thinned_samples(stan_csv))
      return false;
    return true;
  }

  /**
   * Process first chain: record header, thinned samples,
   * add samples to vector chains.
   */
  void initFromStanCsv(const stan::io::stan_csv& stan_csv) {
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
    initFromStanCsv(stan_csv);
  }

  explicit chainset(const std::vector<stan::io::stan_csv>& stan_csv) {
    if (stan_csv.empty())
      return;
    initFromStanCsv(stan_csv[0]);
    for (size_t i = 1; i < stan_csv.size(); ++i) {
      add(stan_csv[i]);
    }
  }

  inline int num_chains() const { return chains_.size(); }

  inline int num_params() const { return param_names_.size(); }

  inline int num_samples() const { return num_samples_; }

  const std::vector<std::string>& param_names() const { return param_names_; }

  const std::string& param_name(int j) const { return param_names_[j]; }

  int index(const std::string& name) const {
    auto it = std::find(param_names_.begin(), param_names_.end(), name);
    if (it != param_names_.end()) {
      return std::distance(param_names_.begin(), it);
    }
    return -1;
  }

  Eigen::MatrixXd samples(const int index) const {
    Eigen::MatrixXd result(num_samples(), chains_.size());
    for (int i = 0; i < chains_.size(); ++i) {
      result.col(i) = chains_[i].col(index);
    }
    return result;
  }

  Eigen::MatrixXd samples(const std::string& name) const {
    return samples(index(name));
  }

  double mean(const int index) const { return samples(index).mean(); }

  double mean(const std::string& name) const { return mean(index(name)); }

  double variance(const int index) const {
    Eigen::MatrixXd draws = samples(index);
    return (draws.array() - draws.mean()).square().sum() / (draws.size() - 1);
  }    

  double variance(const std::string& name) const {
    return variance(index(name));
  }

  double sd(const int index) const { return std::sqrt(variance(index)); }

  double sd(const std::string& name) const { return sd(index(name)); }

  double max_abs_deviation(const int index) const {
    Eigen::MatrixXd draws = samples(index);
    return (samples(index).array() - mean(index)).abs().maxCoeff();
  }    

  double max_abs_deviation(const std::string& name) const {
    return max_abs_deviation(index(name));
  }

  double quantile(const int index, const double prob) const {
    // Ensure the probability is within [0, 1]
    if (prob < 0.0 || prob > 1.0) {
      throw std::out_of_range("Probability must be between 0 and 1.");
    }
    Eigen::MatrixXd draws = samples(index);
    std::vector<double> sorted(draws.data(),
				       draws.data() + draws.size());
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(prob * (sorted.size() - 1));
    return sorted[idx];
  }

  double quantile(const std::string& name, const double prob) const {
    return quantile(index(name), prob);
  }

  double median(const int index) const {
    return quantile(index, .50);
  }

  double median(const std::string& name) const {
    return median(index(name));
  }

  Eigen::VectorXd quantiles(const int index,
                            const Eigen::VectorXd& probs) const {
    // Ensure the probability is within [0, 1]
    if (probs.minCoeff() < 0.0 || probs.maxCoeff() > 1.0) {
      throw std::out_of_range("Probabilities must be between 0 and 1.");
    }
    Eigen::MatrixXd draws = samples(index);
    std::vector<double> sorted(draws.data(),
				       draws.data() + draws.size());
    std::sort(sorted.begin(), sorted.end());
    Eigen::VectorXd quantiles(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
      size_t idx = static_cast<size_t>(probs[i] * (sorted.size() - 1));
      quantiles[i] = sorted[idx];
    }
    return quantiles;
  }

  Eigen::VectorXd quantiles(const std::string& name,
                            const Eigen::VectorXd& probs) const {
    return quantiles(index(name), probs);
  }

  std::pair<double, double> split_rank_normalized_rhat(const int index) const {
    return analyze::split_rank_normalized_rhat(samples(index));
  }

  std::pair<double, double> split_rank_normalized_rhat(
      const std::string& name) const {
    return split_rank_normalized_rhat(index(name));
  }

  std::pair<double, double> split_rank_normalized_ess(const int index) const {
    return analyze::split_rank_normalized_ess(samples(index));
  }

  std::pair<double, double> split_rank_normalized_ess(
      const std::string& name) const {
    return split_rank_normalized_ess(index(name));
  }

  double mcse_mean(const int index) const {
    double ess_bulk = analyze::split_rank_normalized_ess(samples(index)).first;
    return sd(index) / std::sqrt(ess_bulk);
  }

  double mcse_mean(const std::string& name) const {
    return mcse_mean(index(name));
  }

  double mcse_sd(const int index) const {
    Eigen::MatrixXd s = samples(index);
    Eigen::MatrixXd s2 = s.array().square();
    double ess_s = analyze::split_rank_normalized_ess(s).first;
    double ess_s2 = analyze::split_rank_normalized_ess(s2).first;
    return std::min(ess_s, ess_s2);
  }

  double mcse_sd(const std::string& name) const { return mcse_sd(index(name)); }
};

}  // namespace mcmc
}  // namespace stan

#endif
