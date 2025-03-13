#ifndef STAN_SERVICES_UTIL_HMC_NUTS_CONFIG_HPP
#define STAN_SERVICES_UTIL_HMC_NUTS_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/services/util/hmc_nuts_defaults.hpp>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

namespace stan {
namespace services {
namespace util {

/**
 * Configuration class for the NUTS-HMC sampler.
 * 
 * This class holds all configuration parameters for the No-U-Turn Sampler (NUTS)
 * implementation of Hamiltonian Monte Carlo (HMC).
 * 
 * Parameters include settings for the sampler itself and for adaptation.
 * Default values are pulled from stan::services::util::defaults.
 */


class hmc_nuts_config {
 public:

  /**
   * Metric type used by the sampler
   */
  enum class metric_t {
    UNIT_E,
    DIAG_E,
    DENSE_E
  };

  /**
   * Default constructor with default values from stan::services::util::defaults
   */
  hmc_nuts_config() 
    : metric_type_(metric_t::DIAG_E),
      init_inv_metric_(nullptr),
      stepsize_(stepsize::default_value()),
      stepsize_jitter_(stepsize_jitter::default_value()),
      max_depth_(max_depth::default_value()),
      delta_(delta::default_value()),
      gamma_(gamma::default_value()),
      kappa_(kappa::default_value()),
      t0_(t0::default_value()),
      init_buffer_(init_buffer::default_value()),
      term_buffer_(term_buffer::default_value()),
      window_(window::default_value()),
      adaptation_engaged_(adaptation_engaged::default_value()) {
  }

  /**
   * Constructor for a completely specified configuration
   */
  hmc_nuts_config(
      metric_t metric_type,
      double stepsize,
      double stepsize_jitter,
      int max_depth,
      double delta,
      double gamma,
      double kappa,
      double t0,
      unsigned int init_buffer,
      unsigned int term_buffer,
      unsigned int window,
      bool adaptation_engaged,
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr)
    : metric_type_(metric_type),
      init_inv_metric_(init_inv_metric),
      stepsize_(stepsize),
      stepsize_jitter_(stepsize_jitter),
      max_depth_(max_depth),
      delta_(delta),
      gamma_(gamma),
      kappa_(kappa),
      t0_(t0),
      init_buffer_(init_buffer),
      term_buffer_(term_buffer),
      window_(window),
      adaptation_engaged_(adaptation_engaged) {
  }

  /**
   * Configure non-adaptive sampler settings
   */
  void configure_sampler(
      metric_t metric_type,
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value(),
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr) {
    metric_type_ = metric_type;
    init_inv_metric_ = init_inv_metric;
    stepsize_ = stepsize;
    stepsize_jitter_ = stepsize_jitter;
    max_depth_ = max_depth;
    adaptation_engaged_ = false;
  }

  /**
   * Configure settings for adaptive sampler
   */
  void configure_adaptation(
      double delta = delta::default_value(),
      double gamma = gamma::default_value(),
      double kappa = kappa::default_value(),
      double t0 = t0::default_value(),
      unsigned int init_buffer = init_buffer::default_value(),
      unsigned int term_buffer = term_buffer::default_value(),
      unsigned int window = window::default_value()) {
    delta_ = delta;
    gamma_ = gamma;
    kappa_ = kappa;
    t0_ = t0;
    init_buffer_ = init_buffer;
    term_buffer_ = term_buffer;
    window_ = window;
    adaptation_engaged_ = true;
  }

  /**
   * Creates a configuration for a non-adaptive sampler with specified metric
   */
  static hmc_nuts_config non_adaptive(
      metric_t metric_type,
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value(),
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr) {
    hmc_nuts_config config;
    config.configure_sampler(metric_type, stepsize, stepsize_jitter, max_depth, init_inv_metric);
    return config;
  }

  /**
   * Creates a configuration for an adaptive sampler with specified metric
   */
  static hmc_nuts_config adaptive(
      metric_t metric_type,
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value(),
      double delta = delta::default_value(),
      double gamma = gamma::default_value(),
      double kappa = kappa::default_value(),
      double t0 = t0::default_value(),
      unsigned int init_buffer = init_buffer::default_value(),
      unsigned int term_buffer = term_buffer::default_value(),
      unsigned int window = window::default_value(),
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr) {
    hmc_nuts_config config;
    config.configure_sampler(metric_type, stepsize, stepsize_jitter, max_depth, init_inv_metric);
    config.configure_adaptation(delta, gamma, kappa, t0, init_buffer, 
                               term_buffer, window);
    return config;
  }

  /**
   * Validate the configuration parameters, using Stan's default validation functions
   * 
   * @param logger Logger for warning/error messages
   * @return true if configuration is valid
   */
  bool validate(callbacks::logger& logger) const {
    bool valid = true;
    
    // Use the validation functions from the new defaults system
    try {
      stepsize::validate(stepsize_);
    } catch (const std::exception& e) {
      valid = false;
      logger.error(e.what());
    }
    
    try {
      stepsize_jitter::validate(stepsize_jitter_);
    } catch (const std::exception& e) {
      valid = false;
      logger.error(e.what());
    }
    
    try {
      max_depth::validate(max_depth_);
    } catch (const std::exception& e) {
      valid = false;
      logger.error(e.what());
    }
    
    if (adaptation_engaged_) {
      try {
        delta::validate(delta_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        gamma::validate(gamma_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        kappa::validate(kappa_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        t0::validate(t0_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        init_buffer::validate(init_buffer_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        term_buffer::validate(term_buffer_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
      
      try {
        window::validate(window_);
      } catch (const std::exception& e) {
        valid = false;
        logger.error(e.what());
      }
    }
    
    // For DIAG_E and DENSE_E metrics, check if we have an initial inverse
    // metric when one is required
    if (metric_type_ != metric_t::UNIT_E && init_inv_metric_ == nullptr) {
      logger.info("No initial inverse metric provided. "
                  "Using default unit metric.");
    }
    
    return valid;
  }
  
  // Getters
  metric_t metric_type() const { return metric_type_; }
  std::shared_ptr<const stan::io::var_context> init_inv_metric() const { 
    return init_inv_metric_; 
  }
  double stepsize() const { return stepsize_; }
  double stepsize_jitter() const { return stepsize_jitter_; }
  int max_depth() const { return max_depth_; }
  double delta() const { return delta_; }
  double gamma() const { return gamma_; }
  double kappa() const { return kappa_; }
  double t0() const { return t0_; }
  unsigned int init_buffer() const { return init_buffer_; }
  unsigned int term_buffer() const { return term_buffer_; }
  unsigned int window() const { return window_; }
  bool adaptation_engaged() const { return adaptation_engaged_; }
  
  // Setters
  void set_metric_type(metric_t metric_type) { metric_type_ = metric_type; }
  void set_init_inv_metric(std::shared_ptr<const stan::io::var_context> metric) { 
    init_inv_metric_ = metric; 
  }
  void set_stepsize(double stepsize) { stepsize_ = stepsize; }
  void set_stepsize_jitter(double stepsize_jitter) { stepsize_jitter_ = stepsize_jitter; }
  void set_max_depth(int max_depth) { max_depth_ = max_depth; }
  void set_delta(double delta) { delta_ = delta; }
  void set_gamma(double gamma) { gamma_ = gamma; }
  void set_kappa(double kappa) { kappa_ = kappa; }
  void set_t0(double t0) { t0_ = t0; }
  void set_init_buffer(unsigned int init_buffer) { init_buffer_ = init_buffer; }
  void set_term_buffer(unsigned int term_buffer) { term_buffer_ = term_buffer; }
  void set_window(unsigned int window) { window_ = window; }
  void set_adaptation_engaged(bool adaptation_engaged) { adaptation_engaged_ = adaptation_engaged; }


 private:
  metric_t metric_type_;
  std::shared_ptr<const stan::io::var_context> init_inv_metric_;
  double stepsize_;
  double stepsize_jitter_;
  int max_depth_;
  
  // Adaptation parameters
  double delta_;
  double gamma_;
  double kappa_;
  double t0_;
  unsigned int init_buffer_;
  unsigned int term_buffer_;
  unsigned int window_;
  bool adaptation_engaged_;
};

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
