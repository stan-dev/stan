#include "stanrun_c_api.h"

#include <stan/run/load_model.hpp>
#include <stan/run/hmc_nuts.hpp>
#include <stan/run/model_config.hpp>
#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/metric_type.hpp>

#include <memory>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace {

/**
 * Wrapper struct for model handle to ensure type safety
 */
struct model_handle {
  std::unique_ptr<stan::model::model_base> model;
  std::string model_name;
  
  explicit model_handle(stan::model::model_base& m) 
    : model(&m), model_name(m.model_name()) {}
};

/**
 * Copy error message to provided buffer
 */
void copy_error_message(const std::string& error, char* buffer, size_t buffer_size) {
  if (buffer && buffer_size > 0) {
    std::strncpy(buffer, error.c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
  }
}

/**
 * Copy string to provided buffer
 */
void copy_string(const std::string& str, char* buffer, size_t buffer_size) {
  if (buffer && buffer_size > 0) {
    std::strncpy(buffer, str.c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
  }
}

/**
 * Convert string to boolean
 */
bool parse_bool(const std::string& value) {
  std::string lower_value = value;
  std::transform(lower_value.begin(), lower_value.end(), lower_value.begin(), ::tolower);
  
  if (lower_value == "true" || lower_value == "1" || lower_value == "yes") {
    return true;
  } else if (lower_value == "false" || lower_value == "0" || lower_value == "no") {
    return false;
  } else {
    throw std::invalid_argument("Invalid boolean value: " + value);
  }
}

/**
 * Convert string to metric type
 */
stan::run::metric_t parse_metric_type(const std::string& value) {
  if (value == "unit_e") return stan::run::metric_t::UNIT_E;
  if (value == "diag_e") return stan::run::metric_t::DIAG_E;
  if (value == "dense_e") return stan::run::metric_t::DENSE_E;
  throw std::invalid_argument("Invalid metric type: " + value + 
                             ". Valid options: unit_e, diag_e, dense_e");
}

/**
 * Build HMC NUTS configuration from key-value pairs
 */
stan::run::hmc_nuts_config build_hmc_nuts_config(const char* const* keys, 
                                                 const char* const* values, 
                                                 int num_params,
                                                 std::string& actual_output_dir) {
  auto builder = stan::run::hmc_nuts_config::create();
  
  for (int i = 0; i < num_params; ++i) {
    std::string key(keys[i]);
    std::string value(values[i]);
    
    try {
      if (key == "num_chains") {
        builder.num_chains(std::stoul(value));
      } else if (key == "warmup") {
        builder.warmup(std::stoi(value));
      } else if (key == "samples") {
        builder.samples(std::stoi(value));
      } else if (key == "thin") {
        builder.thin(std::stoi(value));
      } else if (key == "refresh") {
        builder.refresh(std::stoi(value));
      } else if (key == "stepsize") {
        builder.stepsize(std::stod(value));
      } else if (key == "stepsize_jitter") {
        builder.stepsize_jitter(std::stod(value));
      } else if (key == "max_depth") {
        builder.max_depth(std::stoi(value));
      } else if (key == "metric_type") {
        builder.metric_type(parse_metric_type(value));
      } else if (key == "delta") {
        builder.delta(std::stod(value));
      } else if (key == "gamma") {
        builder.gamma(std::stod(value));
      } else if (key == "kappa") {
        builder.kappa(std::stod(value));
      } else if (key == "t0") {
        builder.t0(std::stod(value));
      } else if (key == "init_buffer") {
        builder.init_buffer(std::stoul(value));
      } else if (key == "term_buffer") {
        builder.term_buffer(std::stoul(value));
      } else if (key == "window") {
        builder.window(std::stoul(value));
      } else if (key == "output_dir") {
        builder.output_dir(value);
      } else if (key == "seed") {
        builder.seed(std::stoul(value));
      } else if (key == "init_radius") {
        builder.init_radius(std::stod(value));
      } else if (key == "save_start_params") {
        builder.save_start_params(parse_bool(value));
      } else if (key == "save_warmup") {
        builder.save_warmup(parse_bool(value));
      } else if (key == "save_diagnostics") {
        builder.save_diagnostics(parse_bool(value));
      } else if (key == "save_metric") {
        builder.save_metric(parse_bool(value));
      } else {
        throw std::invalid_argument("Unknown parameter: " + key);
      }
    } catch (const std::exception& e) {
      throw std::invalid_argument("Error parsing parameter '" + key + 
                                 "' with value '" + value + "': " + e.what());
    }
  }
  
  auto config = builder.build();
  actual_output_dir = config.output_dir();
  return config;
}

}  // anonymous namespace

extern "C" {

void* stosh_load_model(const char* data_filename, 
                       unsigned int seed,
                       char* error_message, 
                       size_t error_message_size) {
  try {
    auto builder = stan::run::model_config::create().seed(seed);
    
    // Handle optional data file
    if (data_filename && std::strlen(data_filename) > 0) {
      builder.data(data_filename);
    } else {
      builder.data("");  // Empty string - will create empty context
    }
    
    auto config = builder.build();
    auto& model = stan::run::load_model(config);
    
    return new model_handle(model);
    
  } catch (const std::exception& e) {
    copy_error_message("Model loading failed: " + std::string(e.what()), 
                      error_message, error_message_size);
    return nullptr;
  } catch (...) {
    copy_error_message("Unknown error occurred during model loading", 
                      error_message, error_message_size);
    return nullptr;
  }
}

int stosh_run_samplers(void* handle_ptr,
                       const char* const* keys,
                       const char* const* values,
                       int num_params,
                       char* output_dir,
                       size_t output_dir_size,
                       char* error_message,
                       size_t error_message_size) {
  try {
    if (!handle_ptr) {
      copy_error_message("Invalid model handle. Call stosh_load_model first.", 
                        error_message, error_message_size);
      return STANRUN_ERROR_INVALID_ARGS;
    }
    
    if ((num_params > 0) && (!keys || !values)) {
      copy_error_message("Invalid parameter arrays", 
                        error_message, error_message_size);
      return STANRUN_ERROR_INVALID_ARGS;
    }
    
    auto* handle = static_cast<model_handle*>(handle_ptr);
    if (!handle->model) {
      copy_error_message("Invalid model in handle", 
                        error_message, error_message_size);
      return STANRUN_ERROR_INVALID_ARGS;
    }
    
    std::string actual_output_dir;
    auto config = build_hmc_nuts_config(keys, values, num_params, actual_output_dir);
    
    int result = stan::run::hmc_nuts(config, *handle->model);
    
    if (result == 0) {
      // Return the actual output directory used
      copy_string(actual_output_dir, output_dir, output_dir_size);
      return STANRUN_SUCCESS;
    } else {
      copy_error_message("Sampling failed with error code: " + std::to_string(result), 
                        error_message, error_message_size);
      return STANRUN_ERROR_SAMPLING;
    }
    
  } catch (const std::exception& e) {
    copy_error_message("Sampling failed: " + std::string(e.what()), 
                      error_message, error_message_size);
    return STANRUN_ERROR_SAMPLING;
  } catch (...) {
    copy_error_message("Unknown error occurred during sampling", 
                      error_message, error_message_size);
    return STANRUN_ERROR_RUNTIME;
  }
}

void stosh_free_model(void* handle_ptr) {
  if (handle_ptr) {
    delete static_cast<model_handle*>(handle_ptr);
  }
}

const char* stosh_get_model_name(void* handle_ptr) {
  if (handle_ptr) {
    auto* handle = static_cast<model_handle*>(handle_ptr);
    return handle->model_name.c_str();
  }
  return nullptr;
}

}  // extern "C"
