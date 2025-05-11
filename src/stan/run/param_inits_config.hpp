#ifndef STAN_RUN_PARAM_INITS_CONFIG_HPP
#define STAN_RUN_PARAM_INITS_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/run/param_inits_defaults.hpp>
#include <memory>

namespace stan {
namespace run {

/**
 * Configuration class for initial parameter values
 */
class param_inits_config {
public:
  // param_inits_config_builder class embedded as friend
  class param_inits_config_builder {
    friend class param_inits_config;
    stan::run::init_radius init_radius_;
    std::shared_ptr<const stan::io::var_context> init_params_ = nullptr;

  public:
    param_inits_config_builder() : init_radius_() {}

    param_inits_config_builder& init_radius(double radius) {
      init_radius_ = stan::run::init_radius(radius);
      return *this;
    }

    param_inits_config_builder& init_params(
      std::shared_ptr<const stan::io::var_context> params) {
      init_params_ = params;
      return *this;
    }

    param_inits_config build() {
      // validate();
      return param_inits_config(*this);
    }

    void validate() const {
      // check inconsistencies between args
    }
  };

  static param_inits_config_builder create() {
    return param_inits_config_builder();
  }

  double init_radius() const { return init_radius_.value(); }
  std::shared_ptr<const stan::io::var_context> init_params() const {
    return init_params_;
  }

private:
  explicit param_inits_config(const param_inits_config_builder& builder) :
    init_radius_(builder.init_radius_),
    init_params_(builder.init_params_) {
  }

  stan::run::init_radius init_radius_;
  std::shared_ptr<const stan::io::var_context> init_params_;
};

}  // namespace run
}  // namespace stan
#endif
