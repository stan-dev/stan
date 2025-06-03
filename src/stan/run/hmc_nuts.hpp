#ifndef STAN_RUN_HMC_NUTS_HPP
#define STAN_RUN_HMC_NUTS_HPP

#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/hmc_output_writers.hpp>
#include <stan/run/load_samplers.hpp>
#include <stan/run/run_samplers.hpp>
#include <stan/run/metric_type.hpp>
#include <stan/run/read_json_data.hpp>

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <stan/callbacks/json_writer.hpp>

#include <boost/random/mixmax.hpp>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <iostream>

using rng_t = boost::random::mixmax;

namespace stan {
namespace run {

/**
 * Function to run HMC-NUTS algorithm with the specified configuration.
 * 
 * @tparam Jacobian Whether to include Jacobian adjustments
 * @tparam Model The Stan model type
 * @param config HMC-NUTS configuration containing all sampling parameters
 * @param model Reference to the instantiated Stan model
 * @return 0 on success, 1 on error
 */
template <bool Jacobian = true, class Model>
int hmc_nuts(const hmc_nuts_config& config, Model& model) {
  std::stringstream err_msg;
  try {
    stan::callbacks::interrupt interrupt;
    stan::callbacks::stream_logger logger(std::cout, std::cout, std::cout,
                                         std::cerr, std::cerr);
    std::string model_name = model.model_name();

    // Configure outputs
    std::vector<hmc_nuts_writers> writers;
    if (config.num_chains() == 1) {
      auto timestamp = generate_timestamp();
      writers.push_back(create_hmc_nuts_single_chain_writers(
        config, model_name, timestamp, 1));
    } else {
      writers = create_hmc_nuts_multi_chain_writers(config, model_name);
    }

    std::vector<std::string> uparam_names;
    model.unconstrained_param_names(uparam_names, false, false);

    if (uparam_names.empty()) {
      // Handle fixed parameter models
      logger.info("Model has no parameters. Running fixed parameter sampler.");
      logger.info("Fixed parameter model detected - no sampling required.");
      return 0;
    }

    // Initialize parameter contexts - declare outside conditional scope
    std::vector<std::shared_ptr<const stan::io::var_context>> init_contexts;
    init_contexts.reserve(config.num_chains());
    
    if (config.has_init_params()) {
      // Use provided initialization parameters
      init_contexts = config.process().init_params();
    } else {
      // Create empty contexts for default initialization
      for (size_t i = 0; i < config.num_chains(); ++i) {
        init_contexts.push_back(read_json_data(""));
      }
    }

    // Initialize metric contexts
    std::vector<std::shared_ptr<const stan::io::var_context>> metric_contexts;
    metric_contexts.reserve(config.num_chains());
    
    // Get metric contexts from configuration
    auto config_metrics = config.hmc().init_metric();
    if (!config_metrics.empty()) {
      // Use provided metric contexts
      metric_contexts = config_metrics;
    } else {
      // Create empty contexts for default metrics
      for (size_t i = 0; i < config.num_chains(); ++i) {
        metric_contexts.push_back(read_json_data(""));
      }
    }

    // Ensure we have the right number of contexts
    if (init_contexts.size() != config.num_chains()) {
      err_msg << "Mismatch between number of chains (" << config.num_chains() 
              << ") and initialization contexts (" << init_contexts.size() << ")";
      throw std::invalid_argument(err_msg.str());
    }
    
    if (metric_contexts.size() != config.num_chains()) {
      err_msg << "Mismatch between number of chains (" << config.num_chains() 
              << ") and metric contexts (" << metric_contexts.size() << ")";
      throw std::invalid_argument(err_msg.str());
    }

    try {
      run_samplers(model, config, init_contexts, metric_contexts, 
                  writers, interrupt, logger);
    } catch (const std::exception& e) {
      err_msg << "Error running samplers: " << e.what();
      throw std::runtime_error(err_msg.str());
    }

    // Log completion using logger instead of direct console output
    logger.info("Sampling completed successfully!");
    std::stringstream completion_msg;
    completion_msg << "Output dir: " << config.output_dir();
    logger.info(completion_msg.str());
    
    return 0;

  } catch (const std::exception& e) {
    err_msg.str("");  // Clear previous content
    err_msg << "Error: " << e.what();
    std::cerr << err_msg.str() << std::endl;  // Still output to stderr for main program
    return 1;
  }
}

}  // namespace run
}  // namespace stan
#endif
