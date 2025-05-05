#ifndef STAN_RUN_NUTS_ADAPT_DEFAULTS_HPP
#define STAN_RUN_NUTS_ADAPT_DEFAULTS_HPP

#include <stan/run/config/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Adaptation target acceptance statistic.
 */
class delta : public config<double> {
public:
  delta() : config<double>(0.8,
        "Adaptation target acceptance statistic.",
        [](const double& value) {
          if (!(value > 0 && value < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
          }
        }
      ) {}

  explicit delta(double value) : config<double>(value,
        "Adaptation target acceptance statistic.",
        [](const double& value) {
          if (!(value > 0 && value < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
          }
        }
      ) {}

  static double default_value() { return 0.8; }
};

/**
 * Adaptation regularization scale.
 */
class gamma : public config<double> {
public:
  gamma() : config<double>(0.05,
        "Adaptation regularization scale.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("gamma must be greater than 0.");
          }
        }
      ) {}

  explicit gamma(double value) : config<double>(value,
        "Adaptation regularization scale.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("gamma must be greater than 0.");
          }
        }
      ) {}

  static double default_value() { return 0.05; }
};

/**
 * Adaptation relaxation exponent.
 */
class kappa : public config<double> {
public:
  kappa() : config<double>(0.75,
        "Adaptation relaxation exponent.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("kappa must be greater than 0.");
          }
        }
      ) {}

  explicit kappa(double value) : config<double>(value,
        "Adaptation relaxation exponent.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("kappa must be greater than 0.");
          }
        }
      ) {}

  static double default_value() { return 0.75; }
};

/**
 * Adaptation iteration offset.
 */
class t0 : public config<double> {
public:
  t0() : config<double>(10.0,
        "Adaptation iteration offset.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("t0 must be greater than 0.");
          }
        }
      ) {}

  explicit t0(double value) : config<double>(value,
        "Adaptation iteration offset.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("t0 must be greater than 0.");
          }
        }
      ) {}

  static double default_value() { return 10.0; }
};

/**
 * Width of initial fast adaptation interval.
 */
class init_buffer : public config<unsigned int> {
public:
  init_buffer() : config<unsigned int>(75,
        "Width of initial fast adaptation interval."
        // No validator
      ) {}

  explicit init_buffer(unsigned int value) : config<unsigned int>(
        value,
        "Width of initial fast adaptation interval."
        // No validator
      ) {}

  static unsigned int default_value() { return 75; }
};

/**
 * Width of final fast adaptation interval.
 */
class term_buffer : public config<unsigned int> {
public:
  term_buffer() : config<unsigned int>(50,
        "Width of final fast adaptation interval."
        // No validator
      ) {}

  explicit term_buffer(unsigned int value) : config<unsigned int>(value,
        "Width of final fast adaptation interval."
        // No validator
      ) {}

  static unsigned int default_value() { return 50; }
};

/**
 * Initial width of slow adaptation interval.
 */
class window : public config<unsigned int> {
public:
  window() : config<unsigned int>(25,
        "Initial width of slow adaptation interval."
        // No validator
      ) {}

  explicit window(unsigned int value) : config<unsigned int>(value,
        "Initial width of slow adaptation interval."
        // No validator
      ) {}

  static unsigned int default_value() { return 25; }
};

}  // namespace run
}  // namespace stan

#endif
