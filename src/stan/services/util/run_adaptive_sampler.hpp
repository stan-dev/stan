#ifndef STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_RUN_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/blocked_range.h>
#include <ctime>
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
                          callbacks::writer& diagnostic_writer, unsigned int n_chains = 1) {
  Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                          cont_vector.size());

  std::vector<Sampler> all_samps(n_chains, sampler);
  std::vector<double> warmup_times(n_chains, 0);
  std::vector<double> sampler_times(n_chains, 0);
  services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);
  for (int i = 0; i < all_samps.size(); i++) {
    auto& samplerr = all_samps[i];
    samplerr.engage_adaptation();
    try {
      samplerr.z().q = cont_params;
      samplerr.init_stepsize(logger);
    } catch (const std::exception& e) {
      logger.info("Exception initializing step size.");
      logger.info(e.what());
      return;
    }
    stan::mcmc::sample s(cont_params, 0, 0);

    // Headers
    writer.write_sample_names(s, samplerr, model);
    writer.write_diagnostic_names(s, samplerr, model);
    clock_t start = clock();
    util::generate_transitions(samplerr, num_warmup, 0, num_warmup + num_samples,
                               num_thin, refresh, save_warmup, true, writer, s,
                               model, rng, interrupt, logger, i + 1);
    clock_t end = clock();
    double warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    warmup_times.push_back(warm_delta_t);
    samplerr.disengage_adaptation();
    writer.write_adapt_finish(samplerr);
    samplerr.write_sampler_state(sample_writer);

    start = clock();
    util::generate_transitions(samplerr, num_samples, num_warmup,
                               num_warmup + num_samples, num_thin, refresh, true,
                               false, writer, s, model, rng, interrupt, logger, i + 1);
    end = clock();
    double sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    sampler_times.push_back(sample_delta_t);

    writer.write_timing(warm_delta_t, sample_delta_t);
  }
  double accum_warmup_time = stan::math::sum(warmup_times);
  double accum_sample_time = stan::math::sum(sampler_times);
  writer.write_timing(accum_warmup_time, accum_sample_time);

}

template <class Sampler, class Model, class RNG>
void run_adaptive_sampler(Sampler& sampler, Model& model,
                          std::vector<double>& cont_vector, int num_warmup,
                          int num_samples, int num_thin, int refresh,
                          bool save_warmup, RNG& rng,
                          callbacks::interrupt& interrupt,
                          callbacks::logger& logger,
                          std::vector<callbacks::stream_writer>& sample_writer,
                          std::vector<callbacks::stream_writer>& diagnostic_writer, unsigned int n_chains = 1) {
  Eigen::Map<Eigen::VectorXd> cont_params(cont_vector.data(),
                                          cont_vector.size());

  std::vector<Sampler> all_samps(n_chains, sampler);
  std::vector<double> warmup_times(n_chains, 0);
  std::vector<double> sampler_times(n_chains, 0);
  std::vector<services::util::mcmc_writer> writers;
  for (int i = 0; i < all_samps.size(); i++) {
    writers.push_back(services::util::mcmc_writer(sample_writer[i], diagnostic_writer[i], logger));
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n_chains, 1),
   [&](const tbb::blocked_range<size_t>& r) {
    for (size_t i=r.begin(); i!=r.end(); ++i)  {
    // Initialize nested autodiff stack
    stan::mcmc::sample s(cont_params, 0, 0);

    auto& samplerr = all_samps[i];
    auto& writer = writers[i];
    samplerr.engage_adaptation();
    try {
      samplerr.z().q = cont_params;
      samplerr.init_stepsize(logger);
    } catch (const std::exception& e) {
      logger.info("Exception initializing step size.");
      logger.info(e.what());
      return;
    }

    // Headers
    writer.write_sample_names(s, samplerr, model);
    writer.write_diagnostic_names(s, samplerr, model);
    clock_t start = clock();
    util::generate_transitions(samplerr, num_warmup, 0, num_warmup + num_samples,
                               num_thin, refresh, save_warmup, true, writer, s,
                               model, rng, interrupt, logger, i + 1);
    clock_t end = clock();
    double warm_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    warmup_times.push_back(warm_delta_t);
    samplerr.disengage_adaptation();
    writer.write_adapt_finish(samplerr);
    samplerr.write_sampler_state(sample_writer[i]);

    start = clock();
    util::generate_transitions(samplerr, num_samples, num_warmup,
                               num_warmup + num_samples, num_thin, refresh, true,
                               false, writer, s, model, rng, interrupt, logger, i + 1);
    end = clock();
    double sample_delta_t = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    sampler_times.push_back(sample_delta_t);

    writer.write_timing(warm_delta_t, sample_delta_t);
  }
}, tbb::simple_partitioner());

  double accum_warmup_time = stan::math::sum(warmup_times);
  double accum_sample_time = stan::math::sum(sampler_times);
  writers[0].write_timing(accum_warmup_time, accum_sample_time);

}


}  // namespace util
}  // namespace services
}  // namespace stan
#endif
