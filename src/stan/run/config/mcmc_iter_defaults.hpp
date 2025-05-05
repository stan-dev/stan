#ifndef STAN_RUN_MCMC_ITER_DEFAULTS_HPP
#define STAN_RUN_MCMC_ITER_DEFAULTS_HPP

#include <stan/run/config/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Number of warmup iterations.
 */
class num_warmup : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument("num_warmup must be greater than or equal to 0.");
    }
  }

public:
  num_warmup() : config<int>(1000, description_, validator) {}
  
  explicit num_warmup(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1000; }
};

const std::string num_warmup::description_ = "Number of warmup iterations.";

/**
 * Number of sampling iterations.
 */
class num_samples : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value >= 0)) {
      throw std::invalid_argument("num_samples must be greater than or equal to 0.");
    }
  }

public:
  num_samples() : config<int>(1000, description_, validator) {}
  
  explicit num_samples(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1000; }
};

const std::string num_samples::description_ = "Number of sampling iterations.";

/**
 * Save warmup iterations to output.
 */
class save_warmup : public config<bool> {
private:
  static const std::string description_;
  // No validator needed for bool

public:
  save_warmup() : config<bool>(false, description_) {}
  
  explicit save_warmup(bool value) : config<bool>(value, description_) {}

  static bool default_value() { return false; }
};

const std::string save_warmup::description_ = "Save warmup iterations to output.";

/**
 * Period between saved samples.
 */
class thin : public config<int> {
private:
  static const std::string description_;
  static void validator(const int& value) {
    if (!(value > 0)) {
      throw std::invalid_argument("thin must be greater than 0.");
    }
  }

public:
  thin() : config<int>(1, description_, validator) {}
  
  explicit thin(int value) : config<int>(value, description_, validator) {}

  static int default_value() { return 1; }
};

const std::string thin::description_ = "Period between saved samples.";

/**
 * Period between status output messages.
 */
class refresh : public config<int> {
private:
  static const std::string description_;
  // No validator - any refresh value is valid, even negative

public:
  refresh() : config<int>(100, description_) {}
  
  explicit refresh(int value) : config<int>(value, description_) {}

  static int default_value() { return 100; }
};

const std::string refresh::description_ = "Period between status output messages.";

}  // namespace run
}  // namespace stan

#endif
