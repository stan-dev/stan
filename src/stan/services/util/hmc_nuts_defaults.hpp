#ifndef STAN_SERVICES_UTIL_HMC_NUTS_DEFAULTS_HPP
#define STAN_SERVICES_UTIL_HMC_NUTS_DEFAULTS_HPP

#include <stan/services/util/defaults.hpp>
#include <cmath>

namespace stan {
namespace services {
namespace util {

/**
 * Defines the set of default values for all NUTS-HMC sampler options.
 */

// Tag types are empty structs used to distinguish options of the same type
struct stepsize_tag {};
struct stepsize_jitter_tag {};
struct max_depth_tag {};
struct delta_tag {};
struct gamma_tag {};
struct kappa_tag {};
struct t0_tag {};
struct init_buffer_tag {};
struct term_buffer_tag {};
struct window_tag {};
struct adaptation_engaged_tag {};
struct num_warmup_tag {};
struct num_samples_tag {};
struct save_warmup_tag {};
struct thin_tag {};
struct refresh_tag {};
struct int_time_tag {};

// Specializations for each option
// These provide the default values, descriptions, and validation logic

// Stepsize option
template <>
class option<double, stepsize_tag> {
 public:
  static std::string description() {
    return "Step size for discrete evolution.";
  }

  static double default_value() {
    return 1.0;
  }

  static void validate(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("stepsize must be greater than 0.");
    }
  }
};

// Stepsize jitter option
template <>
class option<double, stepsize_jitter_tag> {
 public:
  static std::string description() {
    return "Uniformly random jitter of the stepsize, in percent.";
  }

  static double default_value() {
    return 0.0;
  }

  static void validate(const double& value) {
    if (!(value >= 0 && value <= 1)) {
      throw std::invalid_argument(
          "stepsize_jitter must be between 0 and 1.");
    }
  }
};

// Maximum tree depth option
template <>
class option<int, max_depth_tag> {
 public:
  static std::string description() {
    return "Maximum tree depth.";
  }

  static int default_value() {
    return 10;
  }

  static void validate(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("max_depth must be greater than 0.");
    }
  }
};

// Delta option (target acceptance rate)
template <>
class option<double, delta_tag> {
 public:
  static std::string description() {
    return "Adaptation target acceptance statistic.";
  }

  static double default_value() {
    return 0.8;
  }

  static void validate(const double& value) {
    if (!(value > 0 && value < 1)) {
      throw std::invalid_argument(
          "delta must be between 0 and 1 (exclusive).");
    }
  }
};

// Gamma option (adaptation regularization scale)
template <>
class option<double, gamma_tag> {
 public:
  static std::string description() {
    return "Adaptation regularization scale.";
  }

  static double default_value() {
    return 0.05;
  }

  static void validate(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("gamma must be greater than 0.");
    }
  }
};

// Kappa option (adaptation relaxation exponent)
template <>
class option<double, kappa_tag> {
 public:
  static std::string description() {
    return "Adaptation relaxation exponent.";
  }

  static double default_value() {
    return 0.75;
  }

  static void validate(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("kappa must be greater than 0.");
    }
  }
};

// T0 option (adaptation iteration offset)
template <>
class option<double, t0_tag> {
 public:
  static std::string description() {
    return "Adaptation iteration offset.";
  }

  static double default_value() {
    return 10.0;
  }

  static void validate(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("t0 must be greater than 0.");
    }
  }
};

// Initial buffer width option
template <>
class option<unsigned int, init_buffer_tag> {
 public:
  static std::string description() {
    return "Width of initial fast adaptation interval.";
  }

  static unsigned int default_value() {
    return 75;
  }

  static void validate(const unsigned int& value) {
    // adaptive sampler performs validation
  }
};

// Terminal buffer width option
template <>
class option<unsigned int, term_buffer_tag> {
 public:
  static std::string description() {
    return "Width of final fast adaptation interval.";
  }

  static unsigned int default_value() {
    return 50;
  }

  static void validate(const unsigned int& value) {
    // adaptive sampler performs validation
  }
};

// Window option (initial width of slow adaptation interval)
template <>
class option<unsigned int, window_tag> {
 public:
  static std::string description() {
    return "Initial width of slow adaptation interval.";
  }

  static unsigned int default_value() {
    return 25;
  }

  static void validate(const unsigned int& value) {
    // adaptive sampler performs validation
  }
};

// Adaptation engaged option
template <>
class option<bool, adaptation_engaged_tag> {
 public:
  static std::string description() {
    return "Indicates whether adaptation is engaged.";
  }

  static bool default_value() {
    return true;
  }

  static void validate(const bool& value) {
    // No validation needed for adaptation_engaged
  }
};

// Number of warmup iterations option
template <>
class option<int, num_warmup_tag> {
 public:
  static std::string description() {
    return "Number of warmup iterations.";
  }

  static int default_value() {
    return 1000;
  }

  static void validate(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument(
          "num_warmup must be greater than or equal to 0.");
    }
  }
};

// Number of sampling iterations option
template <>
class option<int, num_samples_tag> {
 public:
  static std::string description() {
    return "Number of sampling iterations.";
  }

  static int default_value() {
    return 1000;
  }

  static void validate(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument(
          "num_samples must be greater than or equal to 0.");
    }
  }
};

// Save warmup iterations option
template <>
class option<bool, save_warmup_tag> {
 public:
  static std::string description() {
    return "Save warmup iterations to output.";
  }

  static bool default_value() {
    return false;
  }

  static void validate(const bool& value) {
    // No validation needed for save_warmup
  }
};

// Thinning option
template <>
class option<int, thin_tag> {
 public:
  static std::string description() {
    return "Period between saved samples.";
  }

  static int default_value() {
    return 1;
  }

  static void validate(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("thin must be greater than 0.");
    }
  }
};

// Refresh option (controls output)
template <>
class option<int, refresh_tag> {
 public:
  static std::string description() {
    return "Period between status output messages.";
  }

  static int default_value() {
    return 100;
  }

  static void validate(const int& value) {
    // Any refresh value is valid, even negative (disables output)
  }
};

// Integration time option for HMC
template <>
class option<double, int_time_tag> {
 public:
  static std::string description() {
    return "Total integration time for Hamiltonian evolution.";
  }

  static double default_value() {
    return M_PI * 2;
  }

  static void validate(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("int_time must be greater than 0.");
    }
  }
};

// Aliases for easier use
using stepsize = option<double, stepsize_tag>;
using stepsize_jitter = option<double, stepsize_jitter_tag>;
using max_depth = option<int, max_depth_tag>;
using delta = option<double, delta_tag>;
using gamma = option<double, gamma_tag>;
using kappa = option<double, kappa_tag>;
using t0 = option<double, t0_tag>;
using init_buffer = option<unsigned int, init_buffer_tag>;
using term_buffer = option<unsigned int, term_buffer_tag>;
using window = option<unsigned int, window_tag>;
using adaptation_engaged = option<bool, adaptation_engaged_tag>;
using num_warmup = option<int, num_warmup_tag>;
using num_samples = option<int, num_samples_tag>;
using save_warmup = option<bool, save_warmup_tag>;
using thin = option<int, thin_tag>;
using refresh = option<int, refresh_tag>;
using int_time = option<double, int_time_tag>;

}  // namespace util
}  // namespace services
}  // namespace stan

#endif
