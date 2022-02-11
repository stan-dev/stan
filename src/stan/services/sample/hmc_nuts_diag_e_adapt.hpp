#ifndef STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_ADAPT_HPP
#define STAN_SERVICES_SAMPLE_HMC_NUTS_DIAG_E_ADAPT_HPP

#include <stan/math/prim.hpp>
#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/mcmc/hmc/nuts/adapt_diag_e_nuts.hpp>
#include <stan/services/util/run_adaptive_sampler.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <vector>

namespace stan {
namespace services {
namespace sample {

/**
 * Runs HMC with NUTS with adaptation using diagonal Euclidean metric
 * with a pre-specified Euclidean metric.
 *
 * @tparam Model Model class
 * @tparam InitContextPtr A type derived from `stan::io::var_context`
 * @tparam InitMetricContext A type derived from `stan::io::var_context`
 * @tparam SamplerWriter A type derived from `stan::callbacks::writer`
 * @tparam DiagnosticWriter A type derived from `stan::callbacks::writer`
 * @tparam InitWriter A type derived from `stan::callbacks::writer`
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] init_inv_metric var context exposing an initial diagonal
              inverse Euclidean metric (must be positive definite)
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] num_warmup Number of warmup samples
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] save_warmup Indicates whether to save the warmup iterations
 * @param[in] refresh Controls the output
 * @param[in] stepsize initial stepsize for discrete evolution
 * @param[in] stepsize_jitter uniform random jitter of stepsize
 * @param[in] max_depth Maximum tree depth
 * @param[in] delta adaptation target acceptance statistic
 * @param[in] gamma adaptation regularization scale
 * @param[in] kappa adaptation relaxation exponent
 * @param[in] t0 adaptation iteration offset
 * @param[in] init_buffer width of initial fast adaptation interval
 * @param[in] term_buffer width of final fast adaptation interval
 * @param[in] window initial width of slow adaptation interval
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] sample_writer Writer for draws
 * @param[in,out] diagnostic_writer Writer for diagnostic information
 * @return error_codes::OK if successful
 */
template <typename Model>
int hmc_nuts_diag_e_adapt(
    Model& model, const stan::io::var_context& init,
    const stan::io::var_context& init_inv_metric, unsigned int random_seed,
    unsigned int chain, double init_radius, int num_warmup, int num_samples,
    int num_thin, bool save_warmup, int refresh, double stepsize,
    double stepsize_jitter, int max_depth, double delta, double gamma,
    double kappa, double t0, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, callbacks::interrupt& interrupt,
    callbacks::logger& logger, callbacks::writer& init_writer,
    callbacks::writer& sample_writer, callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<double> cont_vector = util::initialize(
      model, init, rng, init_radius, true, logger, init_writer);

  Eigen::VectorXd inv_metric;
  try {
    inv_metric = util::read_diag_inv_metric(init_inv_metric,
                                            model.num_params_r(), logger);
    util::validate_diag_inv_metric(inv_metric, logger);
  } catch (const std::domain_error& e) {
    return error_codes::CONFIG;
  }

  stan::mcmc::adapt_diag_e_nuts<Model, boost::ecuyer1988> sampler(model, rng);

  sampler.set_metric(inv_metric);
  sampler.set_nominal_stepsize(stepsize);
  sampler.set_stepsize_jitter(stepsize_jitter);
  sampler.set_max_depth(max_depth);

  sampler.get_stepsize_adaptation().set_mu(log(10 * stepsize));
  sampler.get_stepsize_adaptation().set_delta(delta);
  sampler.get_stepsize_adaptation().set_gamma(gamma);
  sampler.get_stepsize_adaptation().set_kappa(kappa);
  sampler.get_stepsize_adaptation().set_t0(t0);

  sampler.set_window_params(num_warmup, init_buffer, term_buffer, window,
                            logger);

  util::run_adaptive_sampler(
      sampler, model, cont_vector, num_warmup, num_samples, num_thin, refresh,
      save_warmup, rng, interrupt, logger, sample_writer, diagnostic_writer);

  return error_codes::OK;
}

/**
 * Runs HMC with NUTS with adaptation using diagonal Euclidean metric.
 *
 * @tparam Model Model class
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] init var context for initialization
 * @param[in] random_seed random seed for the random number generator
 * @param[in] chain chain id to advance the pseudo random number generator
 * @param[in] init_radius radius to initialize
 * @param[in] num_warmup Number of warmup samples
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] save_warmup Indicates whether to save the warmup iterations
 * @param[in] refresh Controls the output
 * @param[in] stepsize initial stepsize for discrete evolution
 * @param[in] stepsize_jitter uniform random jitter of stepsize
 * @param[in] max_depth Maximum tree depth
 * @param[in] delta adaptation target acceptance statistic
 * @param[in] gamma adaptation regularization scale
 * @param[in] kappa adaptation relaxation exponent
 * @param[in] t0 adaptation iteration offset
 * @param[in] init_buffer width of initial fast adaptation interval
 * @param[in] term_buffer width of final fast adaptation interval
 * @param[in] window initial width of slow adaptation interval
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer Writer callback for unconstrained inits
 * @param[in,out] sample_writer Writer for draws
 * @param[in,out] diagnostic_writer Writer for diagnostic information
 * @return error_codes::OK if successful
 */
template <typename Model>
int hmc_nuts_diag_e_adapt(
    Model& model, const stan::io::var_context& init, unsigned int random_seed,
    unsigned int chain, double init_radius, int num_warmup, int num_samples,
    int num_thin, bool save_warmup, int refresh, double stepsize,
    double stepsize_jitter, int max_depth, double delta, double gamma,
    double kappa, double t0, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, callbacks::interrupt& interrupt,
    callbacks::logger& logger, callbacks::writer& init_writer,
    callbacks::writer& sample_writer, callbacks::writer& diagnostic_writer) {
  stan::io::dump unit_e_metric
      = util::create_unit_e_diag_inv_metric(model.num_params_r());
  return hmc_nuts_diag_e_adapt(
      model, init, unit_e_metric, random_seed, chain, init_radius, num_warmup,
      num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter,
      max_depth, delta, gamma, kappa, t0, init_buffer, term_buffer, window,
      interrupt, logger, init_writer, sample_writer, diagnostic_writer);
}

/**
 * Runs multiple chains of HMC with NUTS with adaptation using diagonal
 * Euclidean metric with a pre-specified Euclidean metric.
 *
 * @tparam Model Model class
 * @tparam InitContextPtr A pointer with underlying type derived from
 `stan::io::var_context`
 * @tparam InitInvContextPtr A pointer with underlying type derived from
 `stan::io::var_context`
 * @tparam SamplerWriter A type derived from `stan::callbacks::writer`
 * @tparam DiagnosticWriter A type derived from `stan::callbacks::writer`
 * @tparam InitWriter A type derived from `stan::callbacks::writer`
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] num_chains The number of chains to run in parallel. `init`,
 * `init_inv_metric`, `init_writer`, `sample_writer`, and `diagnostic_writer`
 must
 * be the same length as this value.
 * @param[in] init An std vector of init var contexts for initialization of each
 * chain.
 * @param[in] init_inv_metric An std vector of var contexts exposing an initial
 diagonal inverse Euclidean metric for each chain (must be positive definite)
 * @param[in] random_seed random seed for the random number generator
 * @param[in] init_chain_id first chain id. The pseudo random number generator
 * will advance for each chain by an integer sequence from `init_chain_id` to
 * `init_chain_id + num_chains - 1`
 * @param[in] init_radius radius to initialize
 * @param[in] num_warmup Number of warmup samples
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] save_warmup Indicates whether to save the warmup iterations
 * @param[in] refresh Controls the output
 * @param[in] stepsize initial stepsize for discrete evolution
 * @param[in] stepsize_jitter uniform random jitter of stepsize
 * @param[in] max_depth Maximum tree depth
 * @param[in] delta adaptation target acceptance statistic
 * @param[in] gamma adaptation regularization scale
 * @param[in] kappa adaptation relaxation exponent
 * @param[in] t0 adaptation iteration offset
 * @param[in] init_buffer width of initial fast adaptation interval
 * @param[in] term_buffer width of final fast adaptation interval
 * @param[in] window initial width of slow adaptation interval
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer std vector of Writer callbacks for unconstrained
 * inits of each chain.
 * @param[in,out] sample_writer std vector of Writers for draws of each chain.
 * @param[in,out] diagnostic_writer std vector of Writers for diagnostic
 * information of each chain.
 * @return error_codes::OK if successful
 */
template <class Model, typename InitContextPtr, typename InitInvContextPtr,
          typename InitWriter, typename SampleWriter, typename DiagnosticWriter>
int hmc_nuts_diag_e_adapt(
    Model& model, size_t num_chains, const std::vector<InitContextPtr>& init,
    const std::vector<InitInvContextPtr>& init_inv_metric,
    unsigned int random_seed, unsigned int init_chain_id, double init_radius,
    int num_warmup, int num_samples, int num_thin, bool save_warmup,
    int refresh, double stepsize, double stepsize_jitter, int max_depth,
    double delta, double gamma, double kappa, double t0,
    unsigned int init_buffer, unsigned int term_buffer, unsigned int window,
    callbacks::interrupt& interrupt, callbacks::logger& logger,
    std::vector<InitWriter>& init_writer,
    std::vector<SampleWriter>& sample_writer,
    std::vector<DiagnosticWriter>& diagnostic_writer) {
  if (num_chains == 1) {
    return hmc_nuts_diag_e_adapt(
        model, *init[0], *init_inv_metric[0], random_seed, init_chain_id,
        init_radius, num_warmup, num_samples, num_thin, save_warmup, refresh,
        stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
        init_buffer, term_buffer, window, interrupt, logger, init_writer[0],
        sample_writer[0], diagnostic_writer[0]);
  }
  using sample_t = stan::mcmc::adapt_diag_e_nuts<Model, boost::ecuyer1988>;
  std::vector<boost::ecuyer1988> rngs;
  rngs.reserve(num_chains);
  std::vector<std::vector<double>> cont_vectors;
  cont_vectors.reserve(num_chains);
  std::vector<sample_t> samplers;
  samplers.reserve(num_chains);
  try {
    for (int i = 0; i < num_chains; ++i) {
      rngs.emplace_back(util::create_rng(random_seed, init_chain_id + i));
      cont_vectors.emplace_back(util::initialize(
          model, *init[i], rngs[i], init_radius, true, logger, init_writer[i]));
      samplers.emplace_back(model, rngs[i]);
      Eigen::VectorXd inv_metric = util::read_diag_inv_metric(
          *init_inv_metric[i], model.num_params_r(), logger);
      util::validate_diag_inv_metric(inv_metric, logger);

      samplers[i].set_metric(inv_metric);
      samplers[i].set_nominal_stepsize(stepsize);
      samplers[i].set_stepsize_jitter(stepsize_jitter);
      samplers[i].set_max_depth(max_depth);

      samplers[i].get_stepsize_adaptation().set_mu(log(10 * stepsize));
      samplers[i].get_stepsize_adaptation().set_delta(delta);
      samplers[i].get_stepsize_adaptation().set_gamma(gamma);
      samplers[i].get_stepsize_adaptation().set_kappa(kappa);
      samplers[i].get_stepsize_adaptation().set_t0(t0);
      samplers[i].set_window_params(num_warmup, init_buffer, term_buffer,
                                    window, logger);
    }
  } catch (const std::domain_error& e) {
    return error_codes::CONFIG;
  }
  tbb::parallel_for(tbb::blocked_range<size_t>(0, num_chains, 1),
                    [num_warmup, num_samples, num_thin, refresh, save_warmup,
                     num_chains, init_chain_id, &samplers, &model, &rngs,
                     &interrupt, &logger, &sample_writer, &cont_vectors,
                     &diagnostic_writer](const tbb::blocked_range<size_t>& r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        util::run_adaptive_sampler(
                            samplers[i], model, cont_vectors[i], num_warmup,
                            num_samples, num_thin, refresh, save_warmup,
                            rngs[i], interrupt, logger, sample_writer[i],
                            diagnostic_writer[i], init_chain_id + i,
                            num_chains);
                      }
                    },
                    tbb::simple_partitioner());
  return error_codes::OK;
}

/**
 * Runs multiple chains of HMC with NUTS with adaptation using diagonal
 * Euclidean metric.
 *
 * @tparam Model Model class
 * @tparam InitContextPtr A pointer with underlying type derived from
 * `stan::io::var_context`
 * @tparam SamplerWriter A type derived from `stan::callbacks::writer`
 * @tparam DiagnosticWriter A type derived from `stan::callbacks::writer`
 * @tparam InitWriter A type derived from `stan::callbacks::writer`
 * @param[in] model Input model to test (with data already instantiated)
 * @param[in] num_chains The number of chains to run in parallel. `init`,
 * `init_writer`, `sample_writer`, and `diagnostic_writer` must be the same
 * length as this value.
 * @param[in] init An std vector of init var contexts for initialization of each
 * chain.
 * @param[in] random_seed random seed for the random number generator
 * @param[in] init_chain_id first chain id. The pseudo random number generator
 * will advance by for each chain by an integer sequence from `init_chain_id` to
 * `init_chain_id+num_chains-1`
 * @param[in] init_radius radius to initialize
 * @param[in] num_warmup Number of warmup samples
 * @param[in] num_samples Number of samples
 * @param[in] num_thin Number to thin the samples
 * @param[in] save_warmup Indicates whether to save the warmup iterations
 * @param[in] refresh Controls the output
 * @param[in] stepsize initial stepsize for discrete evolution
 * @param[in] stepsize_jitter uniform random jitter of stepsize
 * @param[in] max_depth Maximum tree depth
 * @param[in] delta adaptation target acceptance statistic
 * @param[in] gamma adaptation regularization scale
 * @param[in] kappa adaptation relaxation exponent
 * @param[in] t0 adaptation iteration offset
 * @param[in] init_buffer width of initial fast adaptation interval
 * @param[in] term_buffer width of final fast adaptation interval
 * @param[in] window initial width of slow adaptation interval
 * @param[in,out] interrupt Callback for interrupts
 * @param[in,out] logger Logger for messages
 * @param[in,out] init_writer std vector of Writer callbacks for unconstrained
 * inits of each chain.
 * @param[in,out] sample_writer std vector of Writers for draws of each chain.
 * @param[in,out] diagnostic_writer std vector of Writers for diagnostic
 * information of each chain.
 * @return error_codes::OK if successful
 */
template <class Model, typename InitContextPtr, typename InitWriter,
          typename SampleWriter, typename DiagnosticWriter>
int hmc_nuts_diag_e_adapt(
    Model& model, size_t num_chains, const std::vector<InitContextPtr>& init,
    unsigned int random_seed, unsigned int init_chain_id, double init_radius,
    int num_warmup, int num_samples, int num_thin, bool save_warmup,
    int refresh, double stepsize, double stepsize_jitter, int max_depth,
    double delta, double gamma, double kappa, double t0,
    unsigned int init_buffer, unsigned int term_buffer, unsigned int window,
    callbacks::interrupt& interrupt, callbacks::logger& logger,
    std::vector<InitWriter>& init_writer,
    std::vector<SampleWriter>& sample_writer,
    std::vector<DiagnosticWriter>& diagnostic_writer) {
  if (num_chains == 1) {
    return hmc_nuts_diag_e_adapt(
        model, *init[0], random_seed, init_chain_id, init_radius, num_warmup,
        num_samples, num_thin, save_warmup, refresh, stepsize, stepsize_jitter,
        max_depth, delta, gamma, kappa, t0, init_buffer, term_buffer, window,
        interrupt, logger, init_writer[0], sample_writer[0],
        diagnostic_writer[0]);
  }
  std::vector<std::unique_ptr<stan::io::dump>> unit_e_metrics;
  unit_e_metrics.reserve(num_chains);
  for (size_t i = 0; i < num_chains; ++i) {
    unit_e_metrics.emplace_back(std::make_unique<stan::io::dump>(
        util::create_unit_e_diag_inv_metric(model.num_params_r())));
  }
  return hmc_nuts_diag_e_adapt(
      model, num_chains, init, unit_e_metrics, random_seed, init_chain_id,
      init_radius, num_warmup, num_samples, num_thin, save_warmup, refresh,
      stepsize, stepsize_jitter, max_depth, delta, gamma, kappa, t0,
      init_buffer, term_buffer, window, interrupt, logger, init_writer,
      sample_writer, diagnostic_writer);
}

}  // namespace sample
}  // namespace services
}  // namespace stan
#endif
