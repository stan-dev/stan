#ifndef STAN_RUN_LOAD_MODEL_HPP
#define STAN_RUN_LOAD_MODEL_HPP

#include <stan/run/model_config.hpp>
#include <stan/io/var_context.hpp>
#include <stan/model/model_base.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>


// Forward declaration for model creation function
// This is defined in each compiled Stan model
extern stan::model::model_base& new_model(stan::io::var_context& data_context, 
                                  unsigned int seed,
                                  std::ostream* msg_stream);

namespace stan {
namespace run {

/* Load and instantiate a Stan model using the provided arguments
 * 
 * @param args Base arguments containing data file and random seed
 * @return Reference to the instantiated model
 * @throws std::invalid_argument if data file cannot be read
 * @throws std::runtime_error if model instantiation fails
 */
stan::model::model_base&
load_model(const model_config& config) {
  std::stringstream err_msg;

  stan::io::var_context* raw_context = const_cast<stan::io::var_context*>(config.data().get());
  auto& model = ::new_model(*raw_context, config.seed(), &err_msg);
  if (!err_msg.str().empty()) {
    throw std::runtime_error("Error in new_model: " + err_msg.str());
  }
  return model;
}

}  // namespace run
}  // namespace stan

#endif
