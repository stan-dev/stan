#ifndef STAN_RUN_PROCESS_CONFIG_HPP
#define STAN_RUN_PROCESS_CONFIG_HPP

#include <stan/run/config_defaults.hpp>
#include <stan/run/read_json_data.hpp>
#include <stan/io/var_context.hpp>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace stan {
namespace run {

/**
 * Configuration class for inference process.
 * 
 * Handles process-level configuration including chains, output,
 * random seed, and parameter initialization settings.
 * 
 * Parameter initialization behavior:
 * - init_radius: Used for random initialization of parameters not 
 *   specified in init files (or all parameters if no files provided)
 * - init_params: Optional files specifying initial values for some/all
 *   parameters. Missing parameters fall back to random initialization.
 */
class process_config {
public:
  /**
   * Builder class for constructing process configuration.
   */
  class process_config_builder {
    friend class process_config;
    std::string output_dir_;
    num_chains_config num_chains_;
    random_seed_config random_seed_;
    init_radius_config init_radius_;
    std::vector<std::string> init_filenames_;
    mutable std::vector<std::shared_ptr<const stan::io::var_context>> init_params_;

  public:
    process_config_builder() :
      num_chains_(), random_seed_(), init_radius_() {}

    process_config_builder& num_chains(size_t num_chains) {
      num_chains_ = num_chains_config(num_chains);
      return *this;
    }

    process_config_builder& output_dir(const std::string& output_dir) {
      output_dir_ = output_dir;
      return *this;
    }

    process_config_builder& seed(unsigned int seed) {
      random_seed_ = random_seed_config(seed);
      return *this;
    }

    process_config_builder& init_radius(double radius) {
      init_radius_ = init_radius_config(radius);
      return *this;
    }

    process_config_builder& init_params(const std::vector<std::string>& filenames) {
      init_filenames_ = filenames;
      return *this;
    }

    process_config build() {
      validate();
      return process_config(*this);
    }
    
    void validate() const {
      // Validate output directory exists and is writable
      // (Implementation would check filesystem permissions)
      
      if (!init_filenames_.empty()) {
	init_params_.reserve(num_chains_.value());
        if (init_filenames_.size() != num_chains_.value()) {
          std::stringstream err_msg;
          err_msg << "Wrong number of initial parameter files, expecting " 
                  << num_chains_.value() << " filenames but found "
                  << init_filenames_.size();
          throw std::invalid_argument(err_msg.str());
        }
        for (size_t i = 0; i < init_filenames_.size(); ++i) {
          try {
            auto init_context = read_json_data(init_filenames_[i]);
            init_params_.push_back(init_context);
          } catch (const std::exception& e) {
            std::stringstream err_msg;
            err_msg << "Error reading initial params file "
                    << init_filenames_[i] << " for chain " << i
                    << ": " << e.what();
            throw std::runtime_error(err_msg.str());
          }
        }
      }
    }
  };

  static process_config_builder create() {
    return process_config_builder();
  }

  // Getters
  size_t num_chains() const { return num_chains_.value(); }
  unsigned int seed() const { return random_seed_.value(); }
  const std::string& output_dir() const { return output_dir_; }
  double init_radius() const { return init_radius_.value(); }
  
  const std::vector<std::shared_ptr<const stan::io::var_context>>& init_params() const {
    return init_params_;
  }
  
  bool has_init_params() const { return !init_params_.empty(); }

private:
  explicit process_config(const process_config_builder& builder) :
    num_chains_(builder.num_chains_),
    output_dir_(builder.output_dir_),
    random_seed_(builder.random_seed_),
    init_radius_(builder.init_radius_),
    init_params_(builder.init_params_) {
  }

  num_chains_config num_chains_;
  random_seed_config random_seed_;
  std::string output_dir_;
  init_radius_config init_radius_;
  std::vector<std::shared_ptr<const stan::io::var_context>> init_params_;
};

}  // namespace run
}  // namespace stan
#endif
