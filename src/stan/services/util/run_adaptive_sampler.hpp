#ifndef STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <tbb/parallel_for.h>
#include <chrono>
#include <vector>

namespace stan {
namespace services {
namespace util {

/**
 * Runs the sampler with adaptation.
 *
 * @tparam Sampler Type of adaptive sampler.
 * @tparam Model Type of model
 * @tparam RNG Type of random number generator
 * @param[in,out] sampler the mcmc sampler to use on the model
 * @param[in] model the model concept to use for computing log probability
 * @param[in] cont_vector initial parameter values
 * @param[in] num_warmup number of warmup draws
 * @param[in] num_samples number of post warmup draws
 * @param[in] num_thin number to thin the draws. Must be greater than
 *   or equal to 1.
 * @param[in] refresh controls output to the <code>logger</code>
 * @param[in] save_warmup indicates whether the warmup draws should be
 *   sent to the sample writer
 * @param[in,out] rng random number generator
 * @param[in,out] interrupt interrupt callback
 * @param[in,out] logger logger for messages
 * @param[in,out] sample_writer writer for draws
 * @param[in,out] diagnostic_writer writer for diagnostic information
 * @param[in] chain_id The id for a given chain.
 * @param[in] num_chains The number of chains used in the program. This
 *  is used in generate transitions to print out the chain number.
 */
template <typename Sampler, typename Model, typename RNG>
void run_adaptive_sampler(Sampler& sampler, Model& model,
                          std::vector<double>& cont_vector, int num_warmup,
                          int num_samples, int num_thin, int refresh,
                          bool save_warmup, RNG& rng,
                          callbacks::interrupt& interrupt,
                          callbacks::logger& logger,
                          callbacks::writer& sample_writer,
                          callbacks::writer& diagnostic_writer,
                          size_t chain_id = 1, size_t num_chains = 1) {
  Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                          cont_vector.size());

  sampler.engage_adaptation();
  try {
    sampler.z().q = cont_params;
    sampler.init_stepsize(logger);
  } catch (const std::exception& e) {
    logger.info("Exception initializing step size.");
    logger.info(e.what());
    return;
  }

  services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);
  stan::mcmc::sample s(cont_params, 0, 0);

  // Headers
  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  auto start_warm = std::chrono::steady_clock::now();
  util::generate_transitions(sampler, num_warmup, 0, num_warmup + num_samples,
                             num_thin, refresh, save_warmup, true, writer, s,
                             model, rng, interrupt, logger, chain_id,
                             num_chains);
  auto end_warm = std::chrono::steady_clock::now();
  double warm_delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_warm - start_warm)
                            .count()
                        / 1000.0;
  sampler.disengage_adaptation();
  writer.write_adapt_finish(sampler);
  sampler.write_sampler_state(sample_writer);

  auto start_sample = std::chrono::steady_clock::now();
  util::generate_transitions(sampler, num_samples, num_warmup,
                             num_warmup + num_samples, num_thin, refresh, true,
                             false, writer, s, model, rng, interrupt, logger,
                             chain_id, num_chains);
  auto end_sample = std::chrono::steady_clock::now();
  double sample_delta_t = std::chrono::duration_cast<std::chrono::milliseconds>(
                              end_sample - start_sample)
                              .count()
                          / 1000.0;
  writer.write_timing(warm_delta_t, sample_delta_t);
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
