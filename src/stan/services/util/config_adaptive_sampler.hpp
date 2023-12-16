#ifndef STAN_SERVICES_UTIL_CONFIG_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_CONFIG_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/structured_writer.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/services/util/generate_transitions.hpp>
#include <stan/services/util/mcmc_writer.hpp>
#include <tbb/parallel_for.h>
#include <chrono>
#include <iostream>
#include <vector>

namespace stan {
namespace services {
namespace util {

/**
 * Configures the nuts adaptive sampler.
 *
 * @tparam Sampler Type of adaptive sampler.
 * @tparam RNG Type of random number generator
 * @param[in,out] sampler the mcmc sampler to use on the model
 * @param[in] init_inv_metric var context exposing an initial diagonal
 *              inverse Euclidean metric (must be positive definite)
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
 */
template <typename Sampler, typename Metric>
void config_adaptive_sampler(
    Sampler& sampler, Metric& metric, double stepsize, double stepsize_jitter,
    int max_depth, double delta, double gamma, double kappa, double t0,
    int num_warmup, unsigned int init_buffer, unsigned int term_buffer,
    unsigned int window, callbacks::logger& logger) {
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
}


}  // namespace util
}  // namespace services
}  // namespace stan
#endif
