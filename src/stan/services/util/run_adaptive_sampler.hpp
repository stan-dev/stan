#ifndef STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
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
 */
template <class Sampler, class Model, class RNG>
void run_adaptive_sampler(Sampler& sampler, Model& model,
                          std::vector<double>& cont_vector, int num_warmup,
                          int num_samples, int num_thin, int refresh,
                          bool save_warmup, RNG& rng,
                          callbacks::interrupt& interrupt,
                          callbacks::logger& logger,
                          callbacks::writer& sample_writer,
                          callbacks::writer& diagnostic_writer) {
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
                             model, rng, interrupt, logger);
  auto end_warm = std::chrono::steady_clock::now();
  std::chrono::duration<double> warm_delta_t = end_warm - start_warm;

  sampler.disengage_adaptation();
  writer.write_adapt_finish(sampler);
  sampler.write_sampler_state(sample_writer);

  auto start_sample = std::chrono::steady_clock::now();
  util::generate_transitions(sampler, num_samples, num_warmup,
                             num_warmup + num_samples, num_thin, refresh, true,
                             false, writer, s, model, rng, interrupt, logger);
  auto end_sample = std::chrono::steady_clock::now();
  std::chrono::duration<double> sample_delta_t = end_sample - start_sample;

  writer.write_timing(warm_delta_t.count(), sample_delta_t.count());
}
}  // namespace util
}  // namespace services
}  // namespace stan
#endif
