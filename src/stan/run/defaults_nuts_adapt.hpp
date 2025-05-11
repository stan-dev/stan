#ifndef STAN_RUN_DEFAULTS_NUTS_ADAPT_HPP
#define STAN_RUN_DEFAULTS_NUTS_ADAPT_HPP

#include <stan/run/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Adaptation target acceptance statistic.
 */
class delta : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0 && value < 1)) {
      throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
    }
  }

public:
  delta() : config<double>(0.8, description_, validator) {}
  
  explicit delta(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.8; }
};

const std::string delta::description_ = "Adaptation target acceptance statistic.";

/**
 * Adaptation regularization scale.
 */
class gamma : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("gamma must be greater than 0.");
    }
  }

public:
  gamma() : config<double>(0.05, description_, validator) {}
  
  explicit gamma(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.05; }
};

const std::string gamma::description_ = "Adaptation regularization scale.";

/**
 * Adaptation relaxation exponent.
 */
class kappa : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("kappa must be greater than 0.");
    }
  }

public:
  kappa() : config<double>(0.75, description_, validator) {}
  
  explicit kappa(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 0.75; }
};

const std::string kappa::description_ = "Adaptation relaxation exponent.";

/**
 * Adaptation iteration offset.
 */
class t0 : public config<double> {
private:
  static const std::string description_;
  static void validator(const double& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("t0 must be greater than 0.");
    }
  }

public:
  t0() : config<double>(10.0, description_, validator) {}
  
  explicit t0(double value) : config<double>(value, description_, validator) {}

  static double default_value() { return 10.0; }
};

const std::string t0::description_ = "Adaptation iteration offset.";

/**
 * Width of initial fast adaptation interval.
 */
class init_buffer : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  init_buffer() : config<unsigned int>(75, description_) {}
  
  explicit init_buffer(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 75; }
};

const std::string init_buffer::description_ = "Width of initial fast adaptation interval.";

/**
 * Width of final fast adaptation interval.
 */
class term_buffer : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  term_buffer() : config<unsigned int>(50, description_) {}
  
  explicit term_buffer(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 50; }
};

const std::string term_buffer::description_ = "Width of final fast adaptation interval.";

/**
 * Initial width of slow adaptation interval.
 */
class window : public config<unsigned int> {
private:
  static const std::string description_;
  // No validator needed

public:
  window() : config<unsigned int>(25, description_) {}
  
  explicit window(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 25; }
};

const std::string window::description_ = "Initial width of slow adaptation interval.";

}  // namespace run
}  // namespace stan

#endif
