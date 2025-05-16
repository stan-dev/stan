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
    stan::run::num_warmup num_warmup_;
    stan::run::num_samples num_samples_;
    stan::run::save_warmup save_warmup_;
    stan::run::thin thin_;
    stan::run::refresh refresh_;
    stan::run::metric_type_config metric_type_;
    std::shared_ptr<const stan::io::var_context> init_inv_metric_ = nullptr;
    stan::run::stepsize stepsize_;
    stan::run::stepsize_jitter stepsize_jitter_;
    stan::run::max_depth max_depth_;

  public:
    config_hmc_builder() : 
      num_warmup_(),
      num_samples_(),
      save_warmup_(),
      thin_(),
      refresh_(),
      metric_type_(),
      stepsize_(),
      stepsize_jitter_(),
      max_depth_() {}

    config_hmc_builder& num_warmup(int warmup) {
      num_warmup_ = stan::run::num_warmup(warmup);
      return *this;
    }
    
    config_hmc_builder& num_samples(int samples) {
      num_samples_ = stan::run::num_samples(samples);
      return *this;
    }
    
    config_hmc_builder& save_warmup(bool save) {
      save_warmup_ = stan::run::save_warmup(save);
      return *this;
    }
    
    config_hmc_builder& thin(int thin_value) {
      thin_ = stan::run::thin(thin_value);
      return *this;
    }
    
    config_hmc_builder& refresh(int refresh_value) {
      refresh_ = stan::run::refresh(refresh_value);
      return *this;
    }

    config_hmc_builder& metric_type(metric_t metric_type) {
      metric_type_ = stan::run::metric_type_config(metric_type);
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
      validate();
      return config_hmc(*this);
    }

    void validate() const {
      if (thin_.value() > num_samples_.value()) {
        std::stringstream msg;
        msg << "thin cannot exceed num_samples, found thin: " << thin_.value()
            << ", num_samples " << num_samples_.value();
        throw std::invalid_argument(msg.str());
      }
    }
  };

  static config_hmc_builder create() {
    return config_hmc_builder();
  }

  // Getters
  int num_warmup() const { return num_warmup_.value(); }
  int num_samples() const { return num_samples_.value(); }
  bool save_warmup() const { return save_warmup_.value(); }
  int thin() const { return thin_.value(); }
  int refresh() const { return refresh_.value(); }
  metric_t metric_type() const { return metric_type_.value(); }
  std::shared_ptr<const stan::io::var_context> init_inv_metric() const {
    return init_inv_metric_;
  }
  double stepsize() const { return stepsize_.value(); }
  double stepsize_jitter() const { return stepsize_jitter_.value(); }
  int max_depth() const { return max_depth_.value(); }

private:
  explicit config_hmc(const config_hmc_builder& builder) : 
    num_warmup_(builder.num_warmup_),
    num_samples_(builder.num_samples_),
    save_warmup_(builder.save_warmup_),
    thin_(builder.thin_),
    refresh_(builder.refresh_),
    metric_type_(builder.metric_type_),
    init_inv_metric_(builder.init_inv_metric_),
    stepsize_(builder.stepsize_),
    stepsize_jitter_(builder.stepsize_jitter_),
    max_depth_(builder.max_depth_) {
  }

  stan::run::num_warmup num_warmup_;
  stan::run::num_samples num_samples_;
  stan::run::save_warmup save_warmup_;
  stan::run::thin thin_;
  stan::run::refresh refresh_;
  stan::run::metric_type_config metric_type_;
  std::shared_ptr<const stan::io::var_context> init_inv_metric_;
  stan::run::stepsize stepsize_;
  stan::run::stepsize_jitter stepsize_jitter_;
  stan::run::max_depth max_depth_;
};

}  // namespace run
}  // namespace stan
#endif
