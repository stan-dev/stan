#ifndef STAN_SERVICES_UTIL_DISPATCHER_FUNCTIONS_HPP
#define STAN_SERVICES_UTIL_DISPATCHER_FUNCTIONS_HPP

#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>
#include <limits>
#include <string>
#include <vector>
#include <iomanip>
#include <iostream>

namespace stan {
namespace services {
namespace util {

/**
 * Logs model output messages from a stringstream.
 *
 * @param[in] ss Stringstream containing messages
 * @param[in,out] logger Logger to write messages to
 */
inline void log_model_messages(const std::stringstream& ss,
                               callbacks::logger& logger) {
  if (ss.str().length() > 0) {
    logger.info(ss);
  }
}

/**
 * Logs an exception message.
 *
 * @param[in] e Exception that was caught
 * @param[in,out] logger Logger to write messages to
 */
inline void log_exception(const std::exception& e, callbacks::logger& logger) {
  logger.info(e.what());
}

/**
 * Logs an iteration message during sampling.
 *
 * @param[in] m Current iteration
 * @param[in] start Start iteration
 * @param[in] finish End iteration
 * @param[in] warmup Whether in warmup phase
 * @param[in] chain_id Chain identifier
 * @param[in] num_chains Number of chains
 * @param[in,out] logger Logger to write messages to
 */
inline void log_iteration(int m, int start, int finish, bool warmup,
                          size_t chain_id, size_t num_chains,
                          callbacks::logger& logger) {
  int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
  std::stringstream message;

  if (num_chains != 1) {
    message << "Chain [" << chain_id << "] ";
  }

  message << "Iteration: ";
  message << std::setw(it_print_width) << m + 1 + start << " / " << finish;
  message << " [" << std::setw(3)
          << static_cast<int>((100.0 * (start + m + 1)) / finish) << "%] ";
  message << (warmup ? " (Warmup)" : " (Sampling)");

  logger.info(message);
}

/**
 * Dispatch sample header consisting of sample and sampler vars,
 * model variables on the constrained scale:
 * parameters, transformed parameters, generated quantities.
 *
 * @tparam Model Model class
 * @param[in] sample A sample object
 * @param[in] sampler The MCMC sampler
 * @param[in] model The model
 * @param[in,out] dispatcher The dispatcher to write to
 */
template <class Model>
void write_sample_header(stan::mcmc::sample& sample,
                         stan::mcmc::base_mcmc& sampler, Model& model,
                         callbacks::dispatcher& dispatcher,
                         size_t& num_model_values) {
  std::vector<std::string> names;
  sample.get_sample_param_names(names);
  sampler.get_sampler_param_names(names);
  size_t offset = names.size();
  model.constrained_param_names(names, true, true);
  num_model_values = names.size() - offset;
  dispatcher.dispatch(callbacks::info_type::SAMPLE, names);
}

/**
 * Dispatch diagnostics header consisting of sample and sampler vars,
 * and model parameters on the unconstrained scale.
 *
 * @tparam Model Model class
 * @param[in] sample A sample object
 * @param[in] sampler The MCMC sampler
 * @param[in] model The model
 * @param[in,out] dispatcher The dispatcher to write to
 */
template <class Model>
void write_diagnostics_header(stan::mcmc::sample& sample,
                              stan::mcmc::base_mcmc& sampler, Model& model,
                              callbacks::dispatcher& dispatcher) {
  std::vector<std::string> names;
  sample.get_sample_param_names(names);
  sampler.get_sampler_param_names(names);
  std::vector<std::string> model_names;
  model.unconstrained_param_names(model_names, false, false);
  sampler.get_sampler_diagnostic_names(model_names, names);
  dispatcher.dispatch(callbacks::info_type::DIAGNOSTIC, names);
}

/**
 * Assemble information from one sampler iteration
 * Consists of:  inference algorithm state and all parameters,
 * transformed parameters, and generated quantities variables.
 *
 * @tparam Model Model class
 * @tparam RNG RNG type
 * @param[in,out] rng Random number generator
 * @param[in] sample A sample object
 * @param[in] sampler The MCMC sampler
 * @param[in] model The model
 * @param[in,out] dispatcher The dispatcher to write to
 * @param[in,out] logger Logger for messages
 * @param[in] num_model_params Number of model parameters
 */
template <class Model, class RNG>
void write_sample(RNG& rng, stan::mcmc::sample& sample,
                  stan::mcmc::base_mcmc& sampler, Model& model,
                  callbacks::dispatcher& dispatcher, callbacks::logger& logger,
                  size_t num_model_values) {
  std::vector<double> values;
  // append inference algo state to output vector 'values'
  sample.get_sample_params(values);    // lp__, accept_stat__
  sampler.get_sampler_params(values);  // stepsize__, energy__

  std::vector<double> model_values;
  std::vector<int> params_i;
  std::stringstream ss;

  try {
    std::vector<double> cont_params(  // value of params, unconstrained
        sample.cont_params().data(),
        sample.cont_params().data() + sample.cont_params().size());
    model.write_array(rng, cont_params, params_i, model_values, true, true,
                      &ss);

    // Log any model messages
    log_model_messages(ss, logger);
  } catch (const std::domain_error& e) {
    // Log model messages and exception
    log_model_messages(ss, logger);
    log_exception(e, logger);
  } catch (const std::exception& e) {
    // Log model messages and exception, then rethrow
    log_model_messages(ss, logger);
    log_exception(e, logger);
    throw;
  }

  if (model_values.size() > 0)
    values.insert(values.end(), model_values.begin(), model_values.end());
  if (model_values.size() < num_model_values)  // pad data table as needed
    values.insert(values.end(), num_model_values - model_values.size(),
                  std::numeric_limits<double>::quiet_NaN());

  dispatcher.dispatch(callbacks::info_type::SAMPLE, values);
}

/**
 * Dispatch diagnostics (sic) for sampler iteration, consisting of
 * sampler state and gradients of the unconstrained model parameters.
 *
 * @param[in] sample A sample object
 * @param[in] sampler The MCMC sampler
 * @param[in,out] dispatcher The dispatcher to write to
 */
void write_diagnostics(stan::mcmc::sample& sample,
                       stan::mcmc::base_mcmc& sampler,
                       callbacks::dispatcher& dispatcher) {
  std::vector<double> values;

  sample.get_sample_params(values);
  sampler.get_sampler_params(values);
  sampler.get_sampler_diagnostics(values);

  dispatcher.dispatch(callbacks::info_type::DIAGNOSTIC, values);
}

/**
 * Writes adaptation finished message to the dispatcher.
 *
 * @param[in,out] dispatcher The dispatcher to write to
 */
void write_adapt_finish(callbacks::dispatcher& dispatcher) {
  dispatcher.dispatch(callbacks::info_type::SAMPLE, "Adaptation terminated");
}

/**
 * Send timing information to the dispatcher and logger.
 *
 * @param[in] warmDeltaT Warmup time in seconds
 * @param[in] sampleDeltaT Sampling time in seconds
 * @param[in,out] dispatcher The dispatcher to write to
 * @param[in,out] logger Logger for messages
 */
void write_timing(double warmDeltaT, double sampleDeltaT,
                  callbacks::dispatcher& dispatcher,
                  callbacks::logger& logger) {
  std::stringstream timing;
  std::string title(" Elapsed Time: ");
  timing << title << warmDeltaT << " seconds (Warm-up)" << std::endl;
  timing << std::string(title.size(), ' ') << sampleDeltaT
         << " seconds (Sampling)" << std::endl;
  timing << std::string(title.size(), ' ') << warmDeltaT + sampleDeltaT
         << " seconds (Total)";

  dispatcher.dispatch(callbacks::info_type::SAMPLE, timing.str());
  dispatcher.dispatch(callbacks::info_type::DIAGNOSTIC, timing.str());

  logger.info(timing.str());
  logger.info("");
}

/**
 * Handles transitions during MCMC sampling, writing results to the dispatcher.
 *
 * @tparam Model Model class
 * @tparam RNG RNG type
 * @param[in,out] sampler MCMC sampler
 * @param[in] num_iterations Number of iterations
 * @param[in] start Start iteration
 * @param[in] finish End iteration
 * @param[in] num_thin Thinning interval
 * @param[in] refresh Refresh rate for messages
 * @param[in] save Whether to save samples
 * @param[in] warmup Whether in warmup phase
 * @param[in,out] dispatcher Dispatcher for output
 * @param[in,out] logger Logger for messages
 * @param[in] num_model_params Number of model parameters
 * @param[in,out] init_s Initial/current sample
 * @param[in] model Model instance
 * @param[in,out] base_rng Random number generator
 * @param[in,out] callback Interrupt callback
 * @param[in] chain_id Chain identifier
 * @param[in] num_chains Number of chains
 */
template <class Model, class RNG>
void generate_transitions(stan::mcmc::base_mcmc& sampler, int num_iterations,
                          int start, int finish, int num_thin, int refresh,
                          bool save, bool warmup,
                          callbacks::dispatcher& dispatcher,
                          callbacks::logger& logger, size_t num_model_params,
                          stan::mcmc::sample& init_s, Model& model,
                          RNG& base_rng, callbacks::interrupt& callback,
                          size_t chain_id = 1, size_t num_chains = 1) {
  for (int m = 0; m < num_iterations; ++m) {
    callback();

    if (refresh > 0
        && (start + m + 1 == finish || m == 0 || (m + 1) % refresh == 0)) {
      log_iteration(m, start, finish, warmup, chain_id, num_chains, logger);
    }

    init_s = sampler.transition(init_s, logger);

    if (save && ((m % num_thin) == 0)) {
      write_sample(base_rng, init_s, sampler, model, dispatcher, logger,
                   num_model_params);
      write_diagnostics(init_s, sampler, dispatcher);
    }
  }
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
