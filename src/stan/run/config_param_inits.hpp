#ifndef STAN_RUN_CONFIG_PARAM_INITS_HPP
#define STAN_RUN_CONFIG_PARAM_INITS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/run/defaults_param_inits.hpp>
#include <memory>

namespace stan {
namespace run {

/**
 * Configuration class for initial parameter values
 */
class config_param_inits {
public:
  // config_param_inits_builder class embedded as friend
  class config_param_inits_builder {
    friend class config_param_inits;
    stan::run::init_radius init_radius_;
    std::shared_ptr<const stan::io::var_context> init_params_ = nullptr;

  public:
    config_param_inits_builder() : init_radius_() {}

    config_param_inits_builder& init_radius(double radius) {
      init_radius_ = stan::run::init_radius(radius);
      return *this;
    }

    config_param_inits_builder& init_params(
      std::shared_ptr<const stan::io::var_context> params) {
      init_params_ = params;
      return *this;
    }

    config_param_inits build() {
      // validate();
      return config_param_inits(*this);
    }

    void validate() const {
      // check inconsistencies between args
    }
  };

  static config_param_inits_builder create() {
    return config_param_inits_builder();
  }

  double init_radius() const { return init_radius_.value(); }
  std::shared_ptr<const stan::io::var_context> init_params() const {
    return init_params_;
  }

private:
  explicit config_param_inits(const config_param_inits_builder& builder) :
    init_radius_(builder.init_radius_),
    init_params_(builder.init_params_) {
  }

  stan::run::init_radius init_radius_;
  std::shared_ptr<const stan::io::var_context> init_params_;
};

}  // namespace run
}  // namespace stan
#endif
