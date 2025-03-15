#ifndef STAN_RUN_MCMC_ITER_DEFAULTS_HPP
#define STAN_RUN_MCMC_ITER_DEFAULTS_HPP

#include <stan/run/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Number of warmup iterations.
 */
class num_warmup : public config<int> {
public:
  num_warmup() 
    : config<int>(
        1000,  // default value
        "Number of warmup iterations.",
        [](const int& value) {  // validator
          if (!(value >= 0)) {
            throw std::invalid_argument("num_warmup must be greater than or equal to 0.");
          }
        }
      ) {}
      
  explicit num_warmup(int value) 
    : config<int>(
        value,
        "Number of warmup iterations.",
        [](const int& value) {
          if (!(value >= 0)) {
            throw std::invalid_argument("num_warmup must be greater than or equal to 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static int default_value() { return 1000; }
};

/**
 * Number of sampling iterations.
 */
class num_samples : public config<int> {
public:
  num_samples() 
    : config<int>(
        1000,  // default value
        "Number of sampling iterations.",
        [](const int& value) {  // validator
          if (!(value >= 0)) {
            throw std::invalid_argument("num_samples must be greater than or equal to 0.");
          }
        }
      ) {}
      
  explicit num_samples(int value) 
    : config<int>(
        value,
        "Number of sampling iterations.",
        [](const int& value) {
          if (!(value >= 0)) {
            throw std::invalid_argument("num_samples must be greater than or equal to 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static int default_value() { return 1000; }
};

/**
 * Save warmup iterations to output.
 */
class save_warmup : public config<bool> {
public:
  save_warmup() 
    : config<bool>(
        false,  // default value
        "Save warmup iterations to output."
        // No validator
      ) {}
      
  explicit save_warmup(bool value) 
    : config<bool>(
        value,
        "Save warmup iterations to output."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static bool default_value() { return false; }
};

/**
 * Period between saved samples.
 */
class thin : public config<int> {
public:
  thin() 
    : config<int>(
        1,  // default value
        "Period between saved samples.",
        [](const int& value) {  // validator
          if (!(value > 0)) {
            throw std::invalid_argument("thin must be greater than 0.");
          }
        }
      ) {}
      
  explicit thin(int value) 
    : config<int>(
        value,
        "Period between saved samples.",
        [](const int& value) {
          if (!(value > 0)) {
            throw std::invalid_argument("thin must be greater than 0.");
          }
        }
      ) {}

  // Static access to default value for use in other constructors
  static int default_value() { return 1; }
};

/**
 * Period between status output messages.
 */
class refresh : public config<int> {
public:
  refresh() 
    : config<int>(
        100,  // default value
        "Period between status output messages."
        // No validator - any refresh value is valid, even negative
      ) {}
      
  explicit refresh(int value) 
    : config<int>(
        value,
        "Period between status output messages."
        // No validator
      ) {}

  // Static access to default value for use in other constructors
  static int default_value() { return 100; }
};


}  // namespace run
}  // namespace stan

#endif
