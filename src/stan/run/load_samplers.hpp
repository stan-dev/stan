#ifndef STAN_RUN_LOAD_SAMPLERS_HPP
#define STAN_RUN_LOAD_SAMPLERS_HPP

#include <stan/run/hmc_nuts_config.hpp>
#include <stan/run/hmc_output_writers.hpp>
#include <stan/run/metric_type.hpp>

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/stream_logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/io/array_var_context.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_dense_e_nuts.hpp>
#include <stan/mcmc/hmc/nuts/adapt_unit_e_nuts.hpp>
#include <stan/model/model_base.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>

#include <boost/random/mixmax.hpp>

#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace stan {
namespace run {

using rng_t = boost::random::mixmax;

/* Sampler configuration struct to hold initialized samplers and context */
template <typename SamplerType>
struct sampler_config {
  std::vector<SamplerType> samplers;
  std::vector<rng_t> rngs;
  std::vector<std::vector<double>> init_params;
};

/* Template specialization for different metric types */
template <metric_t MetricType>
struct sampler_traits {};

template <>
struct sampler_traits<metric_t::DIAG_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_diag_e_nuts<Model, rng_t>;
};

template <>
struct sampler_traits<metric_t::DENSE_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_dense_e_nuts<Model, rng_t>;
};

template <>
struct sampler_traits<metric_t::UNIT_E> {
  template <typename Model>
  using sampler_type = stan::mcmc::adapt_unit_e_nuts<Model, rng_t>;
};

/* Convenience alias for the variant type */
template <typename Model>
using sampler_variant = std::variant<
  sampler_config<typename sampler_traits<metric_t::UNIT_E>::template sampler_type<Model>>,
  sampler_config<typename sampler_traits<metric_t::DIAG_E>::template sampler_type<Model>>,
  sampler_config<typename sampler_traits<metric_t::DENSE_E>::template sampler_type<Model>>
>;

/* Helper function to configure metric based on type */
template <metric_t MetricType, typename SamplerType, typename Model>
void configure_metric(SamplerType& sampler, const Model& model,
                     const stan::io::var_context* metric_context,
                     stan::callbacks::logger& logger) {
  bool read = true;
  if (metric_context) {
    std::vector<std::string> names;
    metric_context->names_r(names);
    if (names.empty()) {
      read = false;
    }
  }
  if constexpr (MetricType == metric_t::DIAG_E) {
    Eigen::VectorXd inv_metric;
    try {
      if (metric_context && read) {
        inv_metric = stan::services::util::read_diag_inv_metric(
          *metric_context, model.num_params_r(), logger);
      } else {
        inv_metric = Eigen::VectorXd::Ones(model.num_params_r());
      }
    } catch (const std::exception& e) {
      logger.warn("Using unit diagonal metric (failed to read provided metric)");
      inv_metric = Eigen::VectorXd::Ones(model.num_params_r());
    }
    stan::services::util::validate_diag_inv_metric(inv_metric, logger);
    sampler.set_metric(inv_metric);
  } else if constexpr (MetricType == metric_t::DENSE_E) {
    Eigen::MatrixXd inv_metric;
    try {
      if (metric_context && read) {
        inv_metric = stan::services::util::read_dense_inv_metric(
          *metric_context, model.num_params_r(), logger);
      } else {
        inv_metric = Eigen::MatrixXd::Identity(model.num_params_r(), model.num_params_r());
      }
    } catch (const std::exception& e) {
      logger.warn("Using identity matrix metric (failed to read provided metric)");
      inv_metric = Eigen::MatrixXd::Identity(model.num_params_r(), model.num_params_r());
    }
    stan::services::util::validate_dense_inv_metric(inv_metric, logger);
    sampler.set_metric(inv_metric);
  } else if constexpr (MetricType == metric_t::UNIT_E) {
    // Unit metric doesn't need explicit setting - it's the default
    // metric_context is ignored for unit metric
  }
}

/* Helper function to configure common sampler parameters */
template <typename SamplerType>
void configure_sampler_basic(SamplerType& sampler, 
                            const hmc_nuts_config& config) {
  sampler.set_nominal_stepsize(config.stepsize());
  sampler.set_stepsize_jitter(config.stepsize_jitter());
  sampler.set_max_depth(config.max_depth());
  
  sampler.get_stepsize_adaptation().set_mu(std::log(10 * config.stepsize()));
  sampler.get_stepsize_adaptation().set_delta(config.delta());
  sampler.get_stepsize_adaptation().set_gamma(config.gamma());
  sampler.get_stepsize_adaptation().set_kappa(config.kappa());
  sampler.get_stepsize_adaptation().set_t0(config.t0());
}

/* Helper function to configure windowed adaptation (only for diag_e and dense_e) */
template <metric_t MetricType, typename SamplerType>
void configure_windowed_adaptation(SamplerType& sampler,
                                  const hmc_nuts_config& config,
                                  stan::callbacks::logger& logger) {
  if constexpr (MetricType == metric_t::DIAG_E || MetricType == metric_t::DENSE_E) {
      sampler.set_window_params(config.warmup(), config.init_buffer(), 
				config.term_buffer(), config.window(), logger);
  }
  // Unit metric doesn't need windowed adaptation
}

/* Load and configure samplers for a specific metric type
 * 
 * @tparam MetricType The metric type enum value
 * @tparam Model The Stan model type
 * @param model Reference to the instantiated model
 * @param config HMC-NUTS sampler configuration
 * @param init_contexts Vector of initialization contexts for each chain
 * @param metric_contexts Vector of metric contexts for each chain
 * @param logger Reference to logger for messages
 * @param init_writers Vector of writers for initialization output (can contain nullptrs)
 * @return sampler_config containing initialized samplers and related data
 */
template <metric_t MetricType, typename Model>
sampler_config<typename sampler_traits<MetricType>::template sampler_type<Model>>
load_samplers(Model& model,
              const hmc_nuts_config& hmc_nuts_config,
              const std::vector<std::shared_ptr<const stan::io::var_context>>& init_contexts,
              const std::vector<std::shared_ptr<const stan::io::var_context>>& metric_contexts,
              stan::callbacks::logger& logger,
              const std::vector<stan::callbacks::writer*>& init_writers) {
  
  using sampler_type = typename sampler_traits<MetricType>::template sampler_type<Model>;
  using config_type = sampler_config<sampler_type>;
  
  config_type config;
  config.samplers.reserve(hmc_nuts_config.num_chains());
  config.rngs.reserve(hmc_nuts_config.num_chains());
  config.init_params.reserve(hmc_nuts_config.num_chains());
  
  try {
    for (size_t i = 0; i < hmc_nuts_config.num_chains(); ++i) {
      // RNGs passed in?
      // Create RNG for this chain
      config.rngs.emplace_back(stan::services::util::create_rng(hmc_nuts_config.seed(), i + 1));
      
      // Initialize parameters
      stan::io::var_context* init_context =
        const_cast<stan::io::var_context*>(init_contexts[i].get());
      
      // Handle case where init_writer might be null
      stan::callbacks::writer dummy_writer;
      stan::callbacks::writer* writer_to_use = 
        (init_writers[i] != nullptr) ? init_writers[i] : &dummy_writer;

      // initialize with timing message == false      
      auto init_params = stan::services::util::initialize(
        model, *init_context, config.rngs[i], hmc_nuts_config.init_radius(), false, logger,  
        *writer_to_use);
      config.init_params.push_back(std::move(init_params));
      
      // Create and configure sampler
      config.samplers.emplace_back(model, config.rngs[i]);
      auto& sampler = config.samplers.back();
      
      // Configure metric
      configure_metric<MetricType>(sampler, model, metric_contexts[i].get(), logger);
      
      // Configure basic sampler parameters
      configure_sampler_basic(sampler, hmc_nuts_config);
      
      // Configure windowed adaptation (only for diag_e and dense_e)
      configure_windowed_adaptation<MetricType>(sampler, hmc_nuts_config, logger);
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("Error configuring samplers: " + std::string(e.what()));
  }
  
  return config;
}

/* Factory function to create samplers based on runtime metric type
 * Returns a std::variant containing the appropriate sampler configuration
 */
template <typename Model>
sampler_variant<Model>
create_samplers(Model& model,
                const hmc_nuts_config& config,
                const std::vector<std::shared_ptr<const stan::io::var_context>>& init_contexts,
                const std::vector<std::shared_ptr<const stan::io::var_context>>& metric_contexts,
                stan::callbacks::logger& logger,
                const std::vector<stan::callbacks::writer*>& init_writers) {
  
  switch (config.metric_type()) {
    case metric_t::UNIT_E:
      return load_samplers<metric_t::UNIT_E>(model, config, init_contexts, 
                                           metric_contexts, logger, init_writers);
    case metric_t::DIAG_E:
      return load_samplers<metric_t::DIAG_E>(model, config, init_contexts,
                                           metric_contexts, logger, init_writers);
    case metric_t::DENSE_E:
      return load_samplers<metric_t::DENSE_E>(model, config, init_contexts,
                                            metric_contexts, logger, init_writers);
    default:
      throw std::runtime_error("Unknown metric type: " + std::to_string(static_cast<int>(config.metric_type())));
  }
}

}  // namespace run
}  // namespace stan
#endif
