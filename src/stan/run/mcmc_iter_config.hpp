#ifndef STAN_RUN_MCMC_ITER_CONFIG_HPP
#define STAN_RUN_MCMC_ITER_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/run/mcmc_iter_defaults.hpp>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

namespace stan {
namespace run {

/**
 * Configuration class for mcmc sampler iterations parameters.
 * 
 * This class holds all configuration parameters related to
 * iteration counts, thinning, and output frequency.
 */
class mcmc_iter_config {
 public:
  /**
   * Default constructor with default values from parameter classes
   */
  mcmc_iter_config() 
    : num_warmup_param_(),
      num_samples_param_(),
      save_warmup_param_(),
      thin_param_(),
      refresh_param_() {
  }

  /**
   * Constructor for a completely specified configuration
   */
  mcmc_iter_config(
      int num_warmup,
      int num_samples,
      bool save_warmup,
      int thin,
      int refresh)
    : num_warmup_param_(num_warmup),
      num_samples_param_(num_samples),
      save_warmup_param_(save_warmup),
      thin_param_(thin),
      refresh_param_(refresh) {
  }

  /**
   * Creates a configuration with specified parameters
   */
  static mcmc_iter_config create(
      int num_warmup = num_warmup::default_value(),
      int num_samples = num_samples::default_value(),
      bool save_warmup = save_warmup::default_value(),
      int thin = thin::default_value(),
      int refresh = refresh::default_value()) {
    
    return mcmc_iter_config(num_warmup, num_samples, save_warmup, thin, refresh);
  }

  /**
   * Check configuration consistency and log any warnings
   * 
   * @param logger Logger for warning/error messages
   * @return true if configuration is consistent
   */
  bool check_consistency(callbacks::logger& logger) const {
    bool consistent = true;
    
    // Check if the total number of iterations seems reasonable
    const int total_iters = num_warmup() + num_samples();
    if (total_iters > 1e6) {
      logger.warn("Total number of iterations is very large (" + 
                 std::to_string(total_iters) + "). This may take a long time to complete.");
    }
    
    // Check if the thin value makes sense given the number of samples
    if (thin() > num_samples() / 10 && num_samples() > 0) {
      logger.warn("Thinning rate (" + std::to_string(thin()) + 
                 ") is large relative to number of samples (" + 
                 std::to_string(num_samples()) + 
                 "). This may result in fewer draws than expected.");
    }
    
    return consistent;
  }
  
  // Getters that return the parameter values
  int num_warmup() const { return num_warmup_param_.value(); }
  int num_samples() const { return num_samples_param_.value(); }
  bool save_warmup() const { return save_warmup_param_.value(); }
  int thin() const { return thin_param_.value(); }
  int refresh() const { return refresh_param_.value(); }
  
  // Parameter description getters
  std::string num_warmup_description() const { return num_warmup_param_.description(); }
  std::string num_samples_description() const { return num_samples_param_.description(); }
  std::string save_warmup_description() const { return save_warmup_param_.description(); }
  std::string thin_description() const { return thin_param_.description(); }
  std::string refresh_description() const { return refresh_param_.description(); }
  
  // Setters
  void set_num_warmup(int num_warmup) { num_warmup_param_.set_value(num_warmup); }
  void set_num_samples(int num_samples) { num_samples_param_.set_value(num_samples); }
  void set_save_warmup(bool save_warmup) { save_warmup_param_.set_value(save_warmup); }
  void set_thin(int thin) { thin_param_.set_value(thin); }
  void set_refresh(int refresh) { refresh_param_.set_value(refresh); }

 private:
  // Parameter objects that encapsulate default values, validation, and descriptions
  stan::run::num_warmup num_warmup_param_;
  stan::run::num_samples num_samples_param_;
  stan::run::save_warmup save_warmup_param_;
  stan::run::thin thin_param_;
  stan::run::refresh refresh_param_;
};

}  // namespace run
}  // namespace stan
#endif
