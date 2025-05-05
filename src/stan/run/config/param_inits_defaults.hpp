#ifndef STAN_RUN_PARAM_INITS_DEFAULTS_HPP
#define STAN_RUN_PARAM_INITS_DEFAULTS_HPP

#include <stan/run/config/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Initial radius for parameter initialization.
 */
class init_radius : public config<double> {
public:
  init_radius()
    : config<double>(
        2.0,  // default value
        "Initial radius for parameter initialization.",
        [](const double& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("init_radius must be greater than 0.");
          }
        }
      ) {}

  explicit init_radius(double value)
    : config<double>(
        value,
        "Initial radius for parameter initialization.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("init_radius must be greater than 0.");
          }
        }
      ) {}

  static double default_value() { return 2.0; }
};

}  // namespace run
}  // namespace stan

#endif
