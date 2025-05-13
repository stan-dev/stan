#ifndef STAN_RUN_DEFAULTS_ALGORITHM_HPP
#define STAN_RUN_DEFAULTS_ALGORITHM_HPP

#include <stan/run/config.hpp>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Default number of chains configuration.
 */
class num_chains : public config<size_t> {
private:
  static const std::string description_;
  static void validator(const size_t& value) {
    if (value < 1) {
      throw std::invalid_argument("num_chains must be at least 1");
    }
  }

public:
  num_chains() : config<size_t>(1, description_, validator) {}
  
  explicit num_chains(size_t value) : config<size_t>(value, description_, validator) {}

  static size_t default_value() { return 1; }
};

const std::string num_chains::description_ = "Number of Markov chains to run.";

/**
 * Random seed for initialization.
 * 
 * This class wraps the random seed parameter with description.
 * Unlike other configuration parameters, there is no default
 * value - the random seed must be explicitly specified.
 */
class random_seed : public config<unsigned int> {
private:
  static const std::string description_;
  static void validator(const unsigned int& value) {
    // All unsigned int values are valid.
  }

public:
  random_seed() : config<unsigned int>(1, description_, validator) {}

  explicit random_seed(unsigned int value) : config<unsigned int>(value, description_) {}

  static unsigned int default_value() { return 1; }
};

const std::string random_seed::description_ = "Random seed for initialization.";

  
/**
 * Inference algorithm type.
 */
class config_algorithm_type : public config<algorithm_t> {
private:
  static const std::string description_;
  static void validator(const algorithm_t& value) {
    // All values in the enum are valid, so no validation needed
  }

public:
  config_algorithm_type() : config<algorithm_t>(algorithm_t::STAN2_HMC, description_, validator) {}

  explicit config_algorithm_type(algorithm_t value) : config<algorithm_t>(value, description_, validator) {}

  static algorithm_t default_value() { return algorithm_t::STAN2_HMC; }
};

const std::string config_algorithm_type::description_ = "Inference algorithm to run.";

}  // namespace run
}  // namespace stan

#endif
