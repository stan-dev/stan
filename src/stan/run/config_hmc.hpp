#ifndef STAN_RUN_CONFIG_HMC_HPP
#define STAN_RUN_CONFIG_HMC_HPP

#include <stan/io/var_context.hpp>
#include <stan/run/metric_type.hpp>
#include <stan/run/defaults_hmc.hpp>
#include <memory>

namespace stan {
namespace run {

class config_hmc {
public:

  // config_hmc_builder class embedded as friend
  class config_hmc_builder {
    friend class config_hmc;
    stan::run::config_metric_type metric_type_;
    std::shared_ptr<const stan::io::var_context> init_inv_metric_ = nullptr;
    stan::run::stepsize stepsize_;
    stan::run::stepsize_jitter stepsize_jitter_;
    stan::run::max_depth max_depth_;

  public:
    config_hmc_builder() : 
      metric_type_(),
      stepsize_(),
      stepsize_jitter_(),
      max_depth_() {}

    config_hmc_builder& metric_type(metric_t metric_type) {
      metric_type_ = stan::run::config_metric_type(metric_type);
      return *this;
    }
    config_hmc_builder& init_inv_metric(std::shared_ptr<const stan::io::var_context> inv_metric) {
      init_inv_metric_ = inv_metric;
      return *this;
    }
    config_hmc_builder& stepsize(double size) {
      stepsize_ = stan::run::stepsize(size);
      return *this;
    }
    config_hmc_builder& stepsize_jitter(double jitter) {
      stepsize_jitter_ = stan::run::stepsize_jitter(jitter);
      return *this;
    }
    config_hmc_builder& max_depth(int depth) {
      max_depth_ = stan::run::max_depth(depth);
      return *this;
    }

    config_hmc build() {
      // validate();
      return config_hmc(*this);
    }

    void validate() const {
      // check inconsistencies between args
    }
  };

  static config_hmc_builder create() {
    return config_hmc_builder();
  }

  // Getters
  metric_t metric_type() const { return metric_type_.value(); }
  std::shared_ptr<const stan::io::var_context> init_inv_metric() const {
    return init_inv_metric_;
  }
  double stepsize() const { return stepsize_.value(); }
  double stepsize_jitter() const { return stepsize_jitter_.value(); }
  int max_depth() const { return max_depth_.value(); }

private:
  explicit config_hmc(const config_hmc_builder& config_hmc_builder) : 
    metric_type_(config_hmc_builder.metric_type_),
    init_inv_metric_(config_hmc_builder.init_inv_metric_),
    stepsize_(config_hmc_builder.stepsize_),
    stepsize_jitter_(config_hmc_builder.stepsize_jitter_),
    max_depth_(config_hmc_builder.max_depth_) {
  }

  stan::run::config_metric_type metric_type_;
  std::shared_ptr<const stan::io::var_context> init_inv_metric_;
  stan::run::stepsize stepsize_;
  stan::run::stepsize_jitter stepsize_jitter_;
  stan::run::max_depth max_depth_;
};

}  // namespace run
}  // namespace stan
#endif
