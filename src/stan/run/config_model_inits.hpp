#ifndef STAN_RUN_CONFIG_MODEL_INITS_HPP
#define STAN_RUN_CONFIG_MODEL_INITS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/run/defaults_model_inits.hpp>
#include <memory>

namespace stan {
namespace run {

/**
 * Configuration class for model instantiation and initialization.
 */
class config_model_inits {
public:
  // config_model_inits_builder class embedded as friend
  class config_model_inits_builder {
    friend class config_model_inits;
    stan::run::init_radius_config init_radius_;
    std::shared_ptr<const stan::io::var_context> input_data_ = nullptr;
    std::shared_ptr<const stan::io::var_context> init_params_ = nullptr;

  public:
    config_model_inits_builder() : init_radius_() {}

    config_model_inits_builder& init_radius(double radius) {
      init_radius_ = stan::run::init_radius_config(radius);
      return *this;
    }

    config_model_inits_builder& init_params(
      std::shared_ptr<const stan::io::var_context> init_params) {
      init_params_ = init_params;
      return *this;
    }

    config_model_inits_builder& input_data(
      std::shared_ptr<const stan::io::var_context> input_data) {
      input_data_ = input_data;
      return *this;
    }

    config_model_inits build() {
      // validate();
      return config_model_inits(*this);
    }

    void validate() const {
      // check inconsistencies between args
    }
  };

  static config_model_inits_builder create() {
    return config_model_inits_builder();
  }

  double init_radius() const { return init_radius_.value(); }
  std::shared_ptr<const stan::io::var_context> init_params() const {
    return init_params_;
  }
  std::shared_ptr<const stan::io::var_context> input_data() const {
    return input_data_;
  }

private:
  explicit config_model_inits(const config_model_inits_builder& builder) :
    init_radius_(builder.init_radius_),
    init_params_(builder.init_params_) {
  }

  stan::run::init_radius_config init_radius_;
  std::shared_ptr<const stan::io::var_context> init_params_;
  std::shared_ptr<const stan::io::var_context> input_data_;
};

}  // namespace run
}  // namespace stan
#endif
