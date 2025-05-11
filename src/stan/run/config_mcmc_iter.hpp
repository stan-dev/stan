#ifndef STAN_RUN_CONFIG_MCMC_ITER_HPP
#define STAN_RUN_CONFIG_MCMC_ITER_HPP

#include <stan/run/defaults_mcmc_iter.hpp>
#include <sstream>

namespace stan {
namespace run {

/**
 * Configuration class for mcmc sampler iterations parameters.
 * 
 * This class holds all configuration parameters related to
 * iteration counts, thinning, and output frequency.
 */
class config_mcmc_iter {
public:
  // config_mcmc_iter_builder class embedded as friend
  class config_mcmc_iter_builder {
    friend class config_mcmc_iter;

    stan::run::num_warmup num_warmup_;
    stan::run::num_samples num_samples_;
    stan::run::save_warmup save_warmup_;
    stan::run::thin thin_;
    stan::run::refresh refresh_;

  public:
    config_mcmc_iter_builder() : 
      num_warmup_(),
      num_samples_(),
      save_warmup_(),
      thin_(),
      refresh_() {}

    config_mcmc_iter_builder& num_warmup(int warmup) {
      num_warmup_ = stan::run::num_warmup(warmup);
      return *this;
    }
    
    config_mcmc_iter_builder& num_samples(int samples) {
      num_samples_ = stan::run::num_samples(samples);
      return *this;
    }
    
    config_mcmc_iter_builder& save_warmup(bool save) {
      save_warmup_ = stan::run::save_warmup(save);
      return *this;
    }
    
    config_mcmc_iter_builder& thin(int thin_value) {
      thin_ = stan::run::thin(thin_value);
      return *this;
    }
    
    config_mcmc_iter_builder& refresh(int refresh_value) {
      refresh_ = stan::run::refresh(refresh_value);
      return *this;
    }

    config_mcmc_iter build() {
      validate();
      return config_mcmc_iter(*this);
    }

    void validate() const {
      // Additional cross-parameter validation
      if (thin_.value() > num_samples_.value()) {
        std::stringstream msg;
        msg << "thin cannot exceed num_samples, found thin: " << thin_.value()
            << ", num_samples " << num_samples_.value();
        throw std::invalid_argument(msg.str());
      }
    }
  };

  static config_mcmc_iter_builder create() {
    return config_mcmc_iter_builder();
  }
  
  // Getters 
  int num_warmup() const { return num_warmup_.value(); }
  int num_samples() const { return num_samples_.value(); }
  bool save_warmup() const { return save_warmup_.value(); }
  int thin() const { return thin_.value(); }
  int refresh() const { return refresh_.value(); }

private:
  explicit config_mcmc_iter(const config_mcmc_iter_builder& builder) : 
    num_warmup_(builder.num_warmup_),
    num_samples_(builder.num_samples_),
    save_warmup_(builder.save_warmup_),
    thin_(builder.thin_),
    refresh_(builder.refresh_) {
  }

  stan::run::num_warmup num_warmup_;
  stan::run::num_samples num_samples_;
  stan::run::save_warmup save_warmup_;
  stan::run::thin thin_;
  stan::run::refresh refresh_;
};

}  // namespace run
}  // namespace stan
#endif
