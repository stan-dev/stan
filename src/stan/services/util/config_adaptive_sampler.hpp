#ifndef STAN_SERVICES_UTIL_CONFIG_ADAPTIVE_SAMPLER_HPP
#define STAN_SERVICES_UTIL_CONFIG_ADAPTIVE_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
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
 * @param[in] inv_metric initial inverse Euclidean metric
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
 * @param[in,out] logger logger for messages
 */
template <typename Sampler, typename Metric>
void config_adaptive_sampler(
    Sampler& sampler, const Metric& inv_metric,
    double stepsize, double stepsize_jitter,
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
