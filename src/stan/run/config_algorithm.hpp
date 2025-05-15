#ifndef STAN_RUN_CONFIG_ALGORITHM_HPP
#define STAN_RUN_CONFIG_ALGORITHM_HPP

#include <stan/run/algorithm_type.hpp>
#include <stan/run/defaults_algorithm.hpp>
#include <stan/callbacks/logger.hpp>
#include <boost/random/mixmax.hpp>
#include <vector>

namespace stan {
namespace run {

/**
 * Configuration class for algorithm selection parameters.
 * 
 * This class encapsulates the choice of inference algorithm,
 * number of chains, and random seed.
 */
class config_algorithm {
public:

  // config_algorithm_builder class embedded as friend
  class config_algorithm_builder {
    friend class config_algorithm;
    stan::run::algorithm_type_config algorithm_type_;
    stan::run::num_chains_config num_chains_;
    stan::run::random_seed_config random_seed_;
    stan::run::logger_config logger_;
    
   public:
    config_algorithm_builder() :
      algorithm_type_(),
      num_chains_(),
      random_seed_(),
      logger_() {}
    
    config_algorithm_builder& algorithm_type(algorithm_t algorithm) {
      algorithm_type_ = stan::run::algorithm_type_config(algorithm);
      return *this;
    }
    
    config_algorithm_builder& num_chains(size_t num_chains) {
      num_chains_ = stan::run::num_chains_config(num_chains);
      return *this;
    }
    
    config_algorithm_builder& random_seed(unsigned int seed) {
      random_seed_ = stan::run::random_seed_config(seed);
      return *this;
    }
    
    config_algorithm_builder& logger(callbacks::logger* logger) {
      logger_ = stan::run::logger_config(logger);
      return *this;
    }
    
    config_algorithm build() {
      // validate();
      return config_algorithm(*this);
    }
    
    void validate() const {
      // check inconsistencies between args
    }
  };
  
  static config_algorithm_builder create() {
    return config_algorithm_builder();
  }
  
  // Getters
  algorithm_t algorithm_type() const { return algorithm_type_.value(); }
  size_t num_chains() const { return num_chains_.value(); }
  unsigned int random_seed() const { return random_seed_.value(); }
  callbacks::logger* logger() const {return logger_.value(); }
  
private:
  explicit config_algorithm(const config_algorithm_builder& builder) :
       algorithm_type_(builder.algorithm_type_),
       num_chains_(builder.num_chains_),
       random_seed_(builder.random_seed_),
       logger_(builder.logger_){
  }
  
  stan::run::algorithm_type_config algorithm_type_;
  stan::run::num_chains_config num_chains_;
  stan::run::random_seed_config random_seed_;
  stan::run::logger_config logger_;
};

}  // namespace run
}  // namespace stan

#endif
