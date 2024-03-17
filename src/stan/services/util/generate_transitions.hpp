#ifndef STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP
#define STAN_SERVICES_UTIL_GENERATE_TRANSITIONS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/info_type.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <string>

namespace stan {
namespace services {
namespace util {

void log_progress(int iter, int start, int finish, bool warmup,
                  size_t chain_id, size_t num_chains,
                  callbacks::logger& logger) {
  int it_print_width = std::ceil(std::log10(static_cast<double>(finish)));
  std::stringstream message;
  if (num_chains != 1) {
    message << "Chain [" << chain_id << "] ";
  }
  message << "Iteration: ";
  message << std::setw(it_print_width) << iter + 1 + start << " / " << finish;
  message << " [" << std::setw(3)
          << static_cast<int>((100.0 * (start + iter + 1)) / finish) << "%] ";
  message << (warmup ? " (Warmup)" : " (Sampling)");
  logger.info(message);
}

template <class Model, class RNG>
void dispatch_sample(callbacks::dispatcher& dispatcher,
                     RNG& base_rng,
                     stan::mcmc::sample& sample,
                     stan::mcmc::base_mcmc& sampler,
                     Model& model,
                     bool warmup,
                     callbacks::logger& logger) {

  std::vector<double> engine_values;
  sample.get_sample_params(engine_values);  // mcmc:  log_prob, accept_stat
  sampler.get_sampler_params(engine_values);  // nuts-specific
  dispatcher.write_flat(callbacks::table_info_type::ALGO_STATE, engine_values);

  std::vector<double> constrained_values;
  std::vector<int> params_i;
  std::stringstream ss_print;
  std::vector<double> cont_params(
      sample.cont_params().data(),
      sample.cont_params().data() + sample.cont_params().size());

  try {
    model.write_array(base_rng, cont_params, params_i, constrained_values,
                      true, true, &ss_print);
  }
  catch (const std::exception& e) {  // log gq exceptions, continue
    if (ss_print.str().length() > 0)
      logger.info(ss_print);
    ss_print.str("");
    logger.info(e.what());
  }
  if (ss_print.str().length() > 0)
    logger.info(ss_print);

  if (warmup) {
    dispatcher.write_flat(callbacks::table_info_type::DRAW_WARMUP, constrained_values);
    dispatcher.write_flat(callbacks::table_info_type::UPARAMS_WARMUP, cont_params);
  } else {
    dispatcher.write_flat(callbacks::table_info_type::DRAW_SAMPLE, constrained_values);
    dispatcher.write_flat(callbacks::table_info_type::UPARAMS_SAMPLE, cont_params);
  }    
}

/**
 * Generates MCMC transitions, writes to mcmc_writer
 *
 * @tparam Model model class
 * @tparam RNG random number generator class
 * @param[in,out] sampler MCMC sampler used to generate transitions
 * @param[in] num_iterations number of MCMC transitions
 * @param[in] start starting iteration number used for printing messages
 * @param[in] finish end iteration number used for printing messages
 * @param[in] num_thin when save is true, a draw will be written to the
 *   mcmc_writer every num_thin iterations
 * @param[in] refresh number of iterations to print a message. If
 *   refresh is zero, iteration number messages will not be printed
 * @param[in] save if save is true, the transitions will be written
 *   to the mcmc_writer. If false, transitions will not be written
 * @param[in] warmup indicates whether these transitions are warmup. Used
 *   for printing iteration number messages
 * @param[in,out] mcmc_writer writer to handle mcmc output
 * @param[in,out] init_s starts as the initial unconstrained parameter
 *   values. When the function completes, this will have the final
 *   iteration's unconstrained parameter values
 * @param[in] model model
 * @param[in,out] base_rng random number generator
 * @param[in,out] callback interrupt callback called once an iteration
 * @param[in,out] logger logger for messages
 * @param[in] chain_id The id of the current chain, used in output.
 * @param[in] num_chains The number of chains used in the program. This
 *  is used in generate transitions to print out the chain number.
 */
template <class Model, class RNG>
void generate_transitions(stan::mcmc::base_mcmc& sampler, int num_iterations,
                          int start, int finish, int num_thin, int refresh,
                          bool save, bool warmup,
                          util::mcmc_writer& mcmc_writer,
                          stan::mcmc::sample& init_s, Model& model,
                          RNG& base_rng, callbacks::interrupt& callback,
                          callbacks::logger& logger, size_t chain_id = 1,
                          size_t num_chains = 1) {
  for (int m = 0; m < num_iterations; ++m) {
    callback();

    if (refresh > 0
        && (start + m + 1 == finish || m == 0 || (m + 1) % refresh == 0)) {
      log_progress(m, start, finish, warmup, chain_id, num_chains, logger);
    }

    init_s = sampler.transition(init_s, logger);

    if (save && ((m % num_thin) == 0)) {
      mcmc_writer.write_sample_params(base_rng, init_s, sampler, model);
      mcmc_writer.write_diagnostic_params(init_s, sampler);
    }
  }
}

/**
 * Generates MCMC transitions, writes to dispatcher
 *
 * @tparam Model model class
 * @tparam RNG random number generator class
 * @param[in,out] sampler MCMC sampler used to generate transitions
 * @param[in] num_iterations number of MCMC transitions
 * @param[in] start starting iteration number used for printing messages
 * @param[in] finish end iteration number used for printing messages
 * @param[in] num_thin when save is true, a draw will be written to the
 *   mcmc_writer every num_thin iterations
 * @param[in] refresh number of iterations to print a message. If
 *   refresh is zero, iteration number messages will not be printed
 * @param[in] warmup indicates whether these transitions are warmup. Used
 *   for printing iteration number messages
 * @param[in,out] dispatcher - sends outputs to appropriate writer
 * @param[in,out] init_s starts as the initial unconstrained parameter
 *   values. When the function completes, this will have the final
 *   iteration's unconstrained parameter values
 * @param[in] model model
 * @param[in,out] base_rng random number generator
 * @param[in,out] callback interrupt callback called once an iteration
 * @param[in,out] logger logger for messages
 * @param[in] chain_id The id of the current chain, used in output.
 * @param[in] num_chains The number of chains used in the program. This
 *  is used in generate transitions to print out the chain number.
 */
template <class Model, class RNG>
void generate_transitions(stan::mcmc::base_mcmc& sampler, int num_iterations,
                          int start, int finish, int num_thin, int refresh,
                          bool warmup, stan::callbacks::dispatcher& dispatcher,
                          stan::mcmc::sample& init_s, Model& model,
                          RNG& base_rng, callbacks::interrupt& callback,
                          callbacks::logger& logger, size_t chain_id = 1,
                          size_t num_chains = 1) {
  for (int m = 0; m < num_iterations; ++m) {
    callback();
    if (refresh > 0
        && (start + m + 1 == finish || m == 0 || (m + 1) % refresh == 0)) {
      log_progress(m, start, finish, warmup, chain_id, num_chains, logger);
    }
    init_s = sampler.transition(init_s, logger);
    if ((m % num_thin) == 0) {
      dispatch_sample(dispatcher, base_rng, init_s, sampler, model, warmup, logger);
    }
  }
}




}  // namespace util
}  // namespace services
}  // namespace stan

#endif
