#ifndef STAN_SERVICES_UTIL_NUTS_HMC_CONFIG_HPP
#define STAN_SERVICES_UTIL_NUTS_HMC_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>
#include <stan/model/prob_grad.hpp>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
namespace services {
namespace util {

/**
 * nuts_hmc_config object holds configuration information
 * for the nuts-hmc sampler
 * 
 * Warmup parameters
 * 
 * delta adaptation target acceptance statistic
 * gamma adaptation regularization scale
 * kappa adaptation relaxation exponent
 * t0 adaptation iteration offset
 * init_buffer width of initial fast adaptation interval
 * term_buffer width of final fast adaptation interval
 * window initial width of slow adaptation interval
 *
 * Warmup, sample parameters
 *
 * stepsize initial stepsize for discrete evolution
 * stepsize_jitter uniform random jitter of stepsize
 * max_depth Maximum tree depth
 *
 */
class nuts_hmc_config {
 public:
  double stepsize_;
  double stepsize_jitter_;
  int max_depth_;
  double delta_;
  double gamma_;
  double kappa_;
  double t0_;
  unsigned int init_buffer_;
  unsigned int term_buffer_;
  unsigned int window_;
  
  /**
   * Configure warmup and sampling parameters
   */
  void configure_adaptive_sampler(double stepsize, double stepsize_jitter,
				  int max_depth, double delta, double gamma,
				  double kappa, double t0,
				  unsigned int init_buffer,
				  unsigned int term_buffer,
				  unsigned int window) {
    stepsize_ = stepsize;
    stepsize_jitter_ = stepsize_jitter;
    max_depth_ = max_depth;
    delta_ = delta;
    gamma_ = gamma;
    kappa_ = kappa;
    t0_ = t0;
    init_buffer_ = init_buffer;
    term_buffer_ = term_buffer;
    window_ = window;
  }

  /**
   * Constructor - sampling config
   */
  void configure_sampler(double stepsize, double stepsize_jitter,
			 int max_depth) {
    stepsize_ = stepsize;
    stepsize_jitter_ = stepsize_jitter;
    max_depth_ = max_depth;
  }
}  // namespace util
}  // namespace services
}  // namespace stan
#endif
