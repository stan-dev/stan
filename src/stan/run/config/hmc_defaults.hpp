#ifndef STAN_RUN_HMC_DEFAULTS_HPP
#define STAN_RUN_HMC_DEFAULTS_HPP

#include <stan/run/metric_type.hpp>
#include <stan/run/config/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Step size for discrete evolution.
 */
class stepsize : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("stepsize must be greater than 0.");
    }
  }

public:
  stepsize() : config<double>(1.0, description_, validator) {}

  explicit stepsize(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 1.0; }
};

const std::string stepsize::description_ = "Step size for discrete evolution.";

/**
 * Uniformly random jitter of the stepsize, in percent.
 */
class stepsize_jitter : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value >= 0 && value <= 1)) {
      throw std::invalid_argument("stepsize_jitter must be between 0 and 1.");
    }
  }

public:
  stepsize_jitter() : config<double>(0.0, description_, validator) {}

  explicit stepsize_jitter(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.0; }
};

const std::string stepsize_jitter::description_ = "Uniformly random jitter of the stepsize, in percent.";

/**
 * Maximum tree depth.
 */
class max_depth : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("max_depth must be greater than 0.");
    }
  }

public:
  max_depth() : config<int>(10, description_, validator) {}

  explicit max_depth(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 10; }
};

const std::string max_depth::description_ = "Maximum tree depth.";

/**
 * Metric type for the Hamiltonian.
 */
class metric_type_config : public config<metric_t> {
private:
  static const std::string description_;
  static void validator(const metric_t& value) {
    // All values in the enum are valid, so no validation needed
  }

public:
  metric_type_config() : config<metric_t>(metric_t::DIAG_E, description_, validator) {}

  explicit metric_type_config(metric_t value) : config<metric_t>(value, description_, validator) {}

  static metric_t default_value() { return metric_t::DIAG_E; }
};

const std::string metric_type_config::description_ = "Type of metric to use in Hamiltonian dynamics.";

}  // namespace run
}  // namespace stan

#endif
