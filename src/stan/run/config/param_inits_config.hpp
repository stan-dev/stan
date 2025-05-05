#ifndef STAN_RUN_PARAM_INITS_CONFIG_HPP
#define STAN_RUN_PARAM_INITS_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/run/config/param_inits_defaults.hpp>

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
    double init_radius_ = init_radius::default_value();
    std::shared_ptr<const stan::io::var_context> init_params_ = nullptr;

   public:
    param_inits_config_builder& init_radius(double radius) {
      init_radius_ = radius;
      return *this;
    }
    param_inits_config_builder& init_params(
      std::shared_ptr<const stan::io::var_context> params) {
      init_params_ = params;
      return *this; }

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

  double init_radius() const { return init_radius_param_.value(); }
  std::shared_ptr<const stan::io::var_context> init_params() const {
    return init_params_;
  }

 private:
  explicit
  param_inits_config(const param_inits_config_builder&
		     param_inits_config_builder) :
    init_radius_(param_inits_config_builder.init_radius_),
    init_params_(param_inits_config_builder.init_params_),

  std::shared_ptr<const stan::io::var_context> init_params_;
  stan::run::init_radius init_radius_;
};

}  // namespace run
}  // namespace stan
#endif
