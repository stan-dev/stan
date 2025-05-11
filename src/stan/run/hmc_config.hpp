#ifndef STAN_RUN_HMC_CONFIG_HPP
#define STAN_RUN_HMC_CONFIG_HPP

#include <stan/io/var_context.hpp>
#include <stan/run/metric_type.hpp>
#include <stan/run/hmc_defaults.hpp>
#include <memory>

namespace stan {
namespace run {

class hmc_config {
public:

  // hmc_config_builder class embedded as friend
  class hmc_config_builder {
    friend class hmc_config;
    stan::run::metric_type_config metric_type_;
    std::shared_ptr<const stan::io::var_context> init_inv_metric_ = nullptr;
    stan::run::stepsize stepsize_;
    stan::run::stepsize_jitter stepsize_jitter_;
    stan::run::max_depth max_depth_;

  public:
    hmc_config_builder() : 
      metric_type_(),
      stepsize_(),
      stepsize_jitter_(),
      max_depth_() {}

    hmc_config_builder& metric_type(metric_t metric_type) {
      metric_type_ = stan::run::metric_type_config(metric_type);
      return *this;
    }
    hmc_config_builder& init_inv_metric(std::shared_ptr<const stan::io::var_context> inv_metric) {
      init_inv_metric_ = inv_metric;
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
      // validate();
      return hmc_config(*this);
    }

    void validate() const {
      // check inconsistencies between args
    }
  };

  static hmc_config_builder create() {
    return hmc_config_builder();
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
  explicit hmc_config(const hmc_config_builder& hmc_config_builder) : 
    metric_type_(hmc_config_builder.metric_type_),
    init_inv_metric_(hmc_config_builder.init_inv_metric_),
    stepsize_(hmc_config_builder.stepsize_),
    stepsize_jitter_(hmc_config_builder.stepsize_jitter_),
    max_depth_(hmc_config_builder.max_depth_) {
  }

  stan::run::metric_type_config metric_type_;
  std::shared_ptr<const stan::io::var_context> init_inv_metric_;
  stan::run::stepsize stepsize_;
  stan::run::stepsize_jitter stepsize_jitter_;
  stan::run::max_depth max_depth_;
};

}  // namespace run
}  // namespace stan
#endif
