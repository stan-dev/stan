#ifndef STAN_RUN_DEFAULTS_PARAM_INITS_HPP
#define STAN_RUN_DEFAULTS_PARAM_INITS_HPP

#include <stan/run/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Initial radius for parameter initialization.
 */
class init_radius_config : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("init_radius must be greater than 0.");
    }
  }

public:
  init_radius_config() : config<double>(2.0, description_, validator) {}

  explicit init_radius_config(double value) :
    config<double>(value, description_, validator) {}

  static double default_value() { return 2.0; }
};

const std::string init_radius_config::description_ =
  "Initial radius for parameter initialization.";

}  // namespace run
}  // namespace stan

#endif
