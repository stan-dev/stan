#ifndef STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP
#define STAN_SERVICES_SAMPLE_FIXED_PARAM_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/math/prim.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <chrono>
#include <vector>

namespace stan {
namespace services {
namespace sample {

/**
 * Runs the fixed parameter sampler.
 *
 * The fixed parameter sampler sets the parameters randomly once
 * on the unconstrained scale, then runs the model for the number
 * of iterations specified with the parameters fixed.
 *
 * @tparam Model Model class
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] refresh Controls the output
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] sample_writer Writer for draws
 * @param[in,out] diagnostic_writer Writer for diagnostic information
 * @return error_codes::OK if successful
 */
template <class Model>
int fixed_param(Model& model, const stan::io::var_context& init,
                unsigned int random_seed, unsigned int chain,
                double init_radius, int num_samples, int num_thin, int refresh,
                callbacks::interrupt& interrupt, callbacks::logger& logger,
                callbacks::writer& init_writer,
                callbacks::writer& sample_writer,
                callbacks::writer& diagnostic_writer) {
  stan::rng_t rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector;

  try {
    cont_vector = util::initialize(model, init, rng, init_radius, false, logger,
                                   init_writer);
  } catch (const std::exception& e) {
    logger.error(e.what());
    return error_codes::CONFIG;
  }

  stan::mcmc::fixed_param_sampler sampler;
  services::util::mcmc_writer writer(sample_writer, diagnostic_writer, logger);
  Eigen::VectorXd cont_params(cont_vector.size());
  for (size_t i = 0; i < cont_vector.size(); i++)
    cont_params[i] = cont_vector[i];
  stan::mcmc::sample s(cont_params, 0, 0);

  // Headers
  writer.write_sample_names(s, sampler, model);
  writer.write_diagnostic_names(s, sampler, model);

  auto start = std::chrono::steady_clock::now();
  try {
    util::generate_transitions(sampler, num_samples, 0, num_samples, num_thin,
                               refresh, true, false, writer, s, model, rng,
                               interrupt, logger);
  } catch (const std::exception& e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }
  auto end = std::chrono::steady_clock::now();
  double sample_delta_t
      = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count()
        / 1000.0;
  writer.write_timing(0.0, sample_delta_t);

  return error_codes::OK;
}

/**
 * Runs the fixed parameter sampler.
 *
 * The fixed parameter sampler sets the parameters randomly once
 * on the unconstrained scale, then runs the model for the number
 * of iterations specified with the parameters fixed.
 *
 * @tparam Model Model class
 * @tparam InitContextPtr A pointer with underlying type derived from
 * `stan::io::var_context`
 * @tparam SamplerWriter A type derived from `stan::callbacks::writer`
 * @tparam DiagnosticWriter A type derived from `stan::callbacks::writer`
 * @tparam InitWriter A type derived from `stan::callbacks::writer`
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] num_chains Number of chains to run
 * @param[in] init var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] refresh Controls the output
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer std vector of Writer callbacks for unconstrained
 * inits of each chain.
 * @param[in,out] sample_writers std vector of Writers for draws of each chain.
 * @param[in,out] diagnostic_writers std vector of Writers for diagnostic
 * information of each chain.
 * @return error_codes::OK if successful
 */
template <typename Model, typename InitContextPtr, typename InitWriter,
          typename SampleWriter, typename DiagnosticWriter>
int fixed_param(Model& model, const std::size_t num_chains,
                const std::vector<InitContextPtr>& init,
                unsigned int random_seed, unsigned int chain,
                double init_radius, int num_samples, int num_thin, int refresh,
                callbacks::interrupt& interrupt, callbacks::logger& logger,
                std::vector<InitWriter>& init_writer,
                std::vector<SampleWriter>& sample_writers,
                std::vector<DiagnosticWriter>& diagnostic_writers) {
  if (num_chains == 1) {
    return fixed_param(model, *init[0], random_seed, chain, init_radius,
                       num_samples, num_thin, refresh, interrupt, logger,
                       init_writer[0], sample_writers[0],
                       diagnostic_writers[0]);
  }
  std::vector<stan::rng_t> rngs;
  std::vector<Eigen::VectorXd> cont_vectors;
  std::vector<util::mcmc_writer> writers;
  std::vector<stan::mcmc::sample> samples;
  std::vector<stan::mcmc::fixed_param_sampler> samplers(num_chains);
  rngs.reserve(num_chains);
  cont_vectors.reserve(num_chains);
  writers.reserve(num_chains);
  samples.reserve(num_chains);
  for (int i = 0; i < num_chains; ++i) {
    rngs.push_back(util::create_rng(random_seed, chain + i));
    auto cont_vector = util::initialize(model, *init[i], rngs[i], init_radius,
                                        false, logger, init_writer[i]);
    cont_vectors.push_back(
        Eigen::Map<Eigen::VectorXd>(cont_vector.data(), cont_vector.size()));
    samples.emplace_back(cont_vectors[i], 0, 0);
    writers.emplace_back(sample_writers[i], diagnostic_writers[i], logger);
    // Headers
    writers[i].write_sample_names(samples[i], samplers[i], model);
    writers[i].write_diagnostic_names(samples[i], samplers[i], model);
  }

  try {
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_chains, 1),
        [&samplers, &writers, &samples, &model, &rngs, &interrupt, &logger,
         num_samples, num_thin, refresh, chain,
         num_chains](const tbb::blocked_range<size_t>& r) {
          for (size_t i = r.begin(); i != r.end(); ++i) {
            auto start = std::chrono::steady_clock::now();
            util::generate_transitions(
                samplers[i], num_samples, 0, num_samples, num_thin, refresh,
                true, false, writers[i], samples[i], model, rngs[i], interrupt,
                logger, chain + i, num_chains);
            auto end = std::chrono::steady_clock::now();
            double sample_delta_t
                = std::chrono::duration_cast<std::chrono::milliseconds>(end
                                                                        - start)
                      .count()
                  / 1000.0;
            writers[i].write_timing(0.0, sample_delta_t);
          }
        },
        tbb::simple_partitioner());
  } catch (const std::exception& e) {
    logger.error(e.what());
    return error_codes::SOFTWARE;
  }
  return error_codes::OK;
}

}  // namespace sample
}  // namespace services
}  // namespace stan
#endif
