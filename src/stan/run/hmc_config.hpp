#ifndef STAN_RUN_HMC_CONFIG_HPP
#define STAN_RUN_HMC_CONFIG_HPP

#include <stan/run/config_defaults.hpp>
#include <stan/run/metric_type.hpp>
#include <stan/run/read_json_data.hpp>
#include <stan/io/var_context.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Configuration class for Hamiltonian Monte Carlo parameters.
 * 
 * Handles sampling iterations, stepsize, metric type, and other
 * HMC-specific settings.
 */
class hmc_config {
public:
  /**
   * Builder class for constructing HMC configuration.
   */
  class hmc_config_builder {
    friend class hmc_config;
    stan::run::num_warmup warmup_;
    stan::run::num_samples samples_;
    stan::run::thin thin_;
    stan::run::refresh refresh_;
    stan::run::metric_type_config metric_type_;
    mutable std::vector<std::string> metric_filenames_;
    mutable std::vector<std::shared_ptr<const stan::io::var_context>> init_metric_;
    stan::run::stepsize stepsize_;
    stan::run::stepsize_jitter stepsize_jitter_;
    stan::run::max_depth max_depth_;

  public:
    hmc_config_builder() : 
      warmup_(),
      samples_(),
      thin_(),
      refresh_(),
      metric_type_(),
      stepsize_(),
      stepsize_jitter_(),
      max_depth_() {}

    hmc_config_builder& warmup(int warmup) {
      warmup_ = stan::run::num_warmup(warmup);
      return *this;
    }
    
    hmc_config_builder& samples(int samples) {
      samples_ = stan::run::num_samples(samples);
      return *this;
    }
    
    hmc_config_builder& thin(int thin_value) {
      thin_ = stan::run::thin(thin_value);
      return *this;
    }
    
    hmc_config_builder& refresh(int refresh_value) {
      refresh_ = stan::run::refresh(refresh_value);
      return *this;
    }

    hmc_config_builder& metric_type(metric_t metric_type) {
      metric_type_ = stan::run::metric_type_config(metric_type);
      return *this;
    }

    hmc_config_builder& init_metric(const std::vector<std::string>& filenames) {
      metric_filenames_ = filenames;
      return *this;
    }

    hmc_config_builder& stepsize(double size) {
      stepsize_ = stan::run::stepsize(size);
      return *this;
    }

    hmc_config_builder& stepsize_jitter(double jitter) {
      stepsize_jitter_ = stan::run::stepsize_jitter(jitter);
      return *this;
    }

    hmc_config_builder& max_depth(int depth) {
      max_depth_ = stan::run::max_depth(depth);
      return *this;
    }

    hmc_config build() {
      return hmc_config(*this);
    }

    /** 
     * Validate init_metric files and create vector of contexts.
     * Current interface allows user to specify either a single
     * or per-chain metric files.  Validation routine creates
     * vector of contexts, either empty or valid.
     */
    void validate(size_t num_chains) const {
      std::string empty_str;
      init_metric_.reserve(num_chains);
      if (metric_filenames_.empty()) {
	for (size_t i = 0; i < num_chains; ++i) {
	  metric_filenames_.push_back(empty_str);
	}
      } else if (metric_filenames_.size() == 1 && num_chains > 1) {
	for (size_t i = 1; i < num_chains; ++i) {
	  metric_filenames_.push_back(metric_filenames_[0]);
	}
      } else if (metric_filenames_.size() > 1 && 
	  metric_filenames_.size() != num_chains) {
	std::stringstream err_msg;
	err_msg << "Wrong number of metric files, expecting " 
		<< num_chains << " filenames but found "
		<< metric_filenames_.size();
	throw std::invalid_argument(err_msg.str());
      }
      for (size_t i = 0; i < metric_filenames_.size(); ++i) {
	try {
	  auto init_context = read_json_data(metric_filenames_[i]);
	  init_metric_.push_back(init_context);
	} catch (const std::exception& e) {
	  std::stringstream err_msg;
	  err_msg << "Error reading metric file "
		  << metric_filenames_[i] << " for chain " << i
		  << ": " << e.what();
	  throw std::runtime_error(err_msg.str());
	}
      }
    }

  };

  static hmc_config_builder create() {
    return hmc_config_builder();
  }

  // Getters
  int warmup() const { return warmup_.value(); }
  int samples() const { return samples_.value(); }
  int thin() const { return thin_.value(); }
  int refresh() const { return refresh_.value(); }
  metric_t metric_type() const { return metric_type_.value(); }
  std::vector<std::shared_ptr<const stan::io::var_context>> init_metric() const {
    return init_metric_;
  }
  double stepsize() const { return stepsize_.value(); }
  double stepsize_jitter() const { return stepsize_jitter_.value(); }
  int max_depth() const { return max_depth_.value(); }

private:
  explicit hmc_config(const hmc_config_builder& builder) : 
    warmup_(builder.warmup_),
    samples_(builder.samples_),
    thin_(builder.thin_),
    refresh_(builder.refresh_),
    metric_type_(builder.metric_type_),
    init_metric_(builder.init_metric_),
    stepsize_(builder.stepsize_),
    stepsize_jitter_(builder.stepsize_jitter_),
    max_depth_(builder.max_depth_) {
  }

  stan::run::num_warmup warmup_;
  stan::run::num_samples samples_;
  stan::run::thin thin_;
  stan::run::refresh refresh_;
  stan::run::metric_type_config metric_type_;
  std::vector<std::shared_ptr<const stan::io::var_context>> init_metric_;
  stan::run::stepsize stepsize_;
  stan::run::stepsize_jitter stepsize_jitter_;
  stan::run::max_depth max_depth_;
};

}  // namespace run
}  // namespace stan
#endif
