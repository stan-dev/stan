#ifndef STAN_RUN_MODEL_CONFIG_HPP
#define STAN_RUN_MODEL_CONFIG_HPP

#include <stan/run/config_defaults.hpp>
#include <stan/run/read_json_data.hpp>
#include <stan/io/var_context.hpp>
#include <memory>
#include <string>
#include <stdexcept>

namespace stan {
namespace run {

/**
 * Configuration class for model instantiation.
 */
class model_config {
public:
  // model_config_builder class embedded as friend
  class model_config_builder {
    friend class model_config;
    std::string filename_;
    mutable std::shared_ptr<const stan::io::var_context> data_;
    stan::run::random_seed_config seed_;

  public:
    model_config_builder& data(const std::string& filename) {
      filename_ = filename;
      return *this;
    }

    model_config_builder& seed(unsigned int seed) {
      seed_ = stan::run::random_seed_config(seed);
      return *this;
    }

    model_config build() {
      validate();
      return model_config(*this);
    }
    
    void validate() const {
      try {
	data_ = read_json_data(filename_);
      } catch (const std::exception& e) {
	std::stringstream err_msg;
	err_msg << "Error reading input data from file " << filename_
		<< ": " << e.what() << std::endl;
	throw std::runtime_error(err_msg.str());
      }
    }
  };

  static model_config_builder create() {
    return model_config_builder();
  }

  // getters
  unsigned int seed() const { return seed_.value(); }
  std::shared_ptr<const stan::io::var_context> data() const {
    return data_;
  }

private:
  explicit model_config(const model_config_builder& builder) :
    seed_(builder.seed_), data_(builder.data_) {}

  stan::run::random_seed_config seed_;
  std::shared_ptr<const stan::io::var_context> data_;
};

}  // namespace run
}  // namespace stan
#endif
