#ifndef STAN_RUN_NUTS_HMC_DEFAULTS_HPP
#define STAN_RUN_NUTS_HMC_DEFAULTS_HPP

#include <stan/run/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Step size for discrete evolution.
 */
class stepsize : public config<double> {
public:
  stepsize() 
    : config<double>(
        1.0,  // default value
        "Step size for discrete evolution.",
        [](const double& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("stepsize must be greater than 0.");
          }
        }
      ) {}
      
  explicit stepsize(double value) 
    : config<double>(
        value,
        "Step size for discrete evolution.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("stepsize must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 1.0; }
};

/**
 * Uniformly random jitter of the stepsize, in percent.
 */
class stepsize_jitter : public config<double> {
public:
  stepsize_jitter() 
    : config<double>(
        0.0,  // default value
        "Uniformly random jitter of the stepsize, in percent.",
        [](const double& value) {  // validator
          if (!(value >= 0 && value <= 1)) {
            throw std::invalid_argument("stepsize_jitter must be between 0 and 1.");
          }
        }
      ) {}
      
  explicit stepsize_jitter(double value) 
    : config<double>(
        value,
        "Uniformly random jitter of the stepsize, in percent.",
        [](const double& value) {
          if (!(value >= 0 && value <= 1)) {
            throw std::invalid_argument("stepsize_jitter must be between 0 and 1.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 0.0; }
};

/**
 * Maximum tree depth.
 */
class max_depth : public config<int> {
public:
  max_depth() 
    : config<int>(
        10,  // default value
        "Maximum tree depth.",
        [](const int& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("max_depth must be greater than 0.");
          }
        }
      ) {}
      
  explicit max_depth(int value) 
    : config<int>(
        value,
        "Maximum tree depth.",
        [](const int& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("max_depth must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static int default_value() { return 10; }
};

/**
 * Adaptation target acceptance statistic.
 */
class delta : public config<double> {
public:
  delta() 
    : config<double>(
        0.8,  // default value
        "Adaptation target acceptance statistic.",
        [](const double& value) {  // validator
          if (!(value > 0 && value < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
          }
        }
      ) {}
      
  explicit delta(double value) 
    : config<double>(
        value,
        "Adaptation target acceptance statistic.",
        [](const double& value) {
          if (!(value > 0 && value < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1 (exclusive).");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 0.8; }
};

/**
 * Adaptation regularization scale.
 */
class gamma : public config<double> {
public:
  gamma() 
    : config<double>(
        0.05,  // default value
        "Adaptation regularization scale.",
        [](const double& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("gamma must be greater than 0.");
          }
        }
      ) {}
      
  explicit gamma(double value) 
    : config<double>(
        value,
        "Adaptation regularization scale.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("gamma must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 0.05; }
};

/**
 * Adaptation relaxation exponent.
 */
class kappa : public config<double> {
public:
  kappa() 
    : config<double>(
        0.75,  // default value
        "Adaptation relaxation exponent.",
        [](const double& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("kappa must be greater than 0.");
          }
        }
      ) {}
      
  explicit kappa(double value) 
    : config<double>(
        value,
        "Adaptation relaxation exponent.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("kappa must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 0.75; }
};

/**
 * Adaptation iteration offset.
 */
class t0 : public config<double> {
public:
  t0() 
    : config<double>(
        10.0,  // default value
        "Adaptation iteration offset.",
        [](const double& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("t0 must be greater than 0.");
          }
        }
      ) {}
      
  explicit t0(double value) 
    : config<double>(
        value,
        "Adaptation iteration offset.",
        [](const double& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("t0 must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static double default_value() { return 10.0; }
};

/**
 * Width of initial fast adaptation interval.
 */
class init_buffer : public config<unsigned int> {
public:
  init_buffer() 
    : config<unsigned int>(
        75,  // default value
        "Width of initial fast adaptation interval."
        // No validator
      ) {}
      
  explicit init_buffer(unsigned int value) 
    : config<unsigned int>(
        value,
        "Width of initial fast adaptation interval."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static unsigned int default_value() { return 75; }
};

/**
 * Width of final fast adaptation interval.
 */
class term_buffer : public config<unsigned int> {
public:
  term_buffer() 
    : config<unsigned int>(
        50,  // default value
        "Width of final fast adaptation interval."
        // No validator
      ) {}
      
  explicit term_buffer(unsigned int value) 
    : config<unsigned int>(
        value,
        "Width of final fast adaptation interval."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static unsigned int default_value() { return 50; }
};

/**
 * Initial width of slow adaptation interval.
 */
class window : public config<unsigned int> {
public:
  window() 
    : config<unsigned int>(
        25,  // default value
        "Initial width of slow adaptation interval."
        // No validator
      ) {}
      
  explicit window(unsigned int value) 
    : config<unsigned int>(
        value,
        "Initial width of slow adaptation interval."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static unsigned int default_value() { return 25; }
};

/**
 * Indicates whether adaptation is engaged.
 */
class adaptation_engaged : public config<bool> {
public:
  adaptation_engaged() 
    : config<bool>(
        true,  // default value
        "Indicates whether adaptation is engaged."
        // No validator
      ) {}
      
  explicit adaptation_engaged(bool value) 
    : config<bool>(
        value,
        "Indicates whether adaptation is engaged."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static bool default_value() { return true; }
};

}  // namespace run
}  // namespace stan

#endif
