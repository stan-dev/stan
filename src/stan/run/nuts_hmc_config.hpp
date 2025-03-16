#ifndef STAN_RUN_NUTS_HMC_CONFIG_HPP
#define STAN_RUN_NUTS_HMC_CONFIG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/io/var_context.hpp>
#include <stan/run/nuts_hmc_defaults.hpp>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

namespace stan {
namespace run {

/**
 * Configuration class for the NUTS-HMC sampler.
 * 
 * This class holds all configuration parameters for the No-U-Turn Sampler (NUTS)
 * implementation of Hamiltonian Monte Carlo (HMC).
 * 
 * Parameters include settings for the sampler itself and for adaptation.
 */
class nuts_hmc_config {
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
   * Default constructor with default values from parameter classes
   */
  nuts_hmc_config() 
    : metric_type_(metric_t::DIAG_E),
      init_inv_metric_(nullptr),
      init_radius_param_(),
      stepsize_param_(),
      stepsize_jitter_param_(),
      max_depth_param_(),
      delta_param_(),
      gamma_param_(),
      kappa_param_(),
      t0_param_(),
      init_buffer_param_(),
      term_buffer_param_(),
      window_param_(),
      adaptation_engaged_param_() {
  }

  /**
   * Constructor for a completely specified configuration
   */
  nuts_hmc_config(
      metric_t metric_type,
      std::shared_ptr<const stan::io::var_context> init_inv_metric,
      double init_radius,
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
      bool adaptation_engaged)
    : metric_type_(metric_type),
      init_inv_metric_(init_inv_metric),
      init_radius_param_(init_radius),
      stepsize_param_(stepsize),
      stepsize_jitter_param_(stepsize_jitter),
      max_depth_param_(max_depth),
      delta_param_(delta),
      gamma_param_(gamma),
      kappa_param_(kappa),
      t0_param_(t0),
      init_buffer_param_(init_buffer),
      term_buffer_param_(term_buffer),
      window_param_(window),
      adaptation_engaged_param_(adaptation_engaged) {
  }

  /**
   * Configure non-adaptive sampler settings
   */
  void configure_sampler(
      metric_t metric_type,
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr,
      double init_radius = init_radius::default_value(),
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value()) {
    metric_type_ = metric_type;
    init_inv_metric_ = init_inv_metric;
    init_radius_param_.set_value(init_radius);
    stepsize_param_.set_value(stepsize);
    stepsize_jitter_param_.set_value(stepsize_jitter);
    max_depth_param_.set_value(max_depth);
    adaptation_engaged_param_.set_value(false);
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
    delta_param_.set_value(delta);
    gamma_param_.set_value(gamma);
    kappa_param_.set_value(kappa);
    t0_param_.set_value(t0);
    init_buffer_param_.set_value(init_buffer);
    term_buffer_param_.set_value(term_buffer);
    window_param_.set_value(window);
    adaptation_engaged_param_.set_value(true);
  }

  /**
   * Creates a configuration for a non-adaptive sampler with specified metric
   */
  static nuts_hmc_config non_adaptive(
      metric_t metric_type,
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr,
      double init_radius = init_radius::default_value(),
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value()) {
    nuts_hmc_config config;
    config.configure_sampler(metric_type, init_inv_metric, init_radius,
			     stepsize, stepsize_jitter, max_depth);
    return config;
  }

  /**
   * Creates a configuration for an adaptive sampler with specified metric
   */
  static nuts_hmc_config adaptive(
      metric_t metric_type,
      std::shared_ptr<const stan::io::var_context> init_inv_metric = nullptr,
      double init_radius = init_radius::default_value(),
      double stepsize = stepsize::default_value(),
      double stepsize_jitter = stepsize_jitter::default_value(),
      int max_depth = max_depth::default_value(),
      double delta = delta::default_value(),
      double gamma = gamma::default_value(),
      double kappa = kappa::default_value(),
      double t0 = t0::default_value(),
      unsigned int init_buffer = init_buffer::default_value(),
      unsigned int term_buffer = term_buffer::default_value(),
      unsigned int window = window::default_value()) {
    nuts_hmc_config config;
    config.configure_sampler(metric_type, init_inv_metric, init_radius,
			     stepsize, stepsize_jitter, max_depth);
    config.configure_adaptation(delta, gamma, kappa, t0, init_buffer, 
                               term_buffer, window);
    return config;
  }

  /**
   * Check configuration consistency and log any warnings
   * 
   * @param logger Logger for warning/error messages
   * @return true if configuration is consistent
   */
  bool check_consistency(callbacks::logger& logger) const {
    bool consistent = true;
    
    // For DIAG_E and DENSE_E metrics, check if we have an initial inverse
    // metric when one is required
    if (metric_type_ != metric_t::UNIT_E && init_inv_metric_ == nullptr) {
      logger.info("No initial inverse metric provided. "
                  "Using default unit metric.");
    }
    
    return consistent;
  }
  
  // Getters that return the parameter values
  metric_t metric_type() const { return metric_type_; }
  std::shared_ptr<const stan::io::var_context> init_inv_metric() const { 
    return init_inv_metric_; 
  }
  double init_radius() const { return init_radius_param_.value(); }
  double stepsize() const { return stepsize_param_.value(); }
  double stepsize_jitter() const { return stepsize_jitter_param_.value(); }
  int max_depth() const { return max_depth_param_.value(); }
  double delta() const { return delta_param_.value(); }
  double gamma() const { return gamma_param_.value(); }
  double kappa() const { return kappa_param_.value(); }
  double t0() const { return t0_param_.value(); }
  unsigned int init_buffer() const { return init_buffer_param_.value(); }
  unsigned int term_buffer() const { return term_buffer_param_.value(); }
  unsigned int window() const { return window_param_.value(); }
  bool adaptation_engaged() const { return adaptation_engaged_param_.value(); }
  
  // Parameter description getters
  std::string init_radius_description() const { return init_radius_param_.description(); }
  std::string stepsize_description() const { return stepsize_param_.description(); }
  std::string stepsize_jitter_description() const { return stepsize_jitter_param_.description(); }
  std::string max_depth_description() const { return max_depth_param_.description(); }
  std::string delta_description() const { return delta_param_.description(); }
  std::string gamma_description() const { return gamma_param_.description(); }
  std::string kappa_description() const { return kappa_param_.description(); }
  std::string t0_description() const { return t0_param_.description(); }
  std::string init_buffer_description() const { return init_buffer_param_.description(); }
  std::string term_buffer_description() const { return term_buffer_param_.description(); }
  std::string window_description() const { return window_param_.description(); }
  std::string adaptation_engaged_description() const { return adaptation_engaged_param_.description(); }
  
  // Setters
  void set_metric_type(metric_t metric_type) { metric_type_ = metric_type; }
  void set_init_inv_metric(std::shared_ptr<const stan::io::var_context> metric) { 
    init_inv_metric_ = metric; 
  }
  void set_stepsize(double stepsize) { stepsize_param_.set_value(stepsize); }
  void set_stepsize_jitter(double stepsize_jitter) { stepsize_jitter_param_.set_value(stepsize_jitter); }
  void set_init_radius(double init_radius) { init_radius_param_.set_value(init_radius); }
  void set_max_depth(int max_depth) { max_depth_param_.set_value(max_depth); }
  void set_delta(double delta) { delta_param_.set_value(delta); }
  void set_gamma(double gamma) { gamma_param_.set_value(gamma); }
  void set_kappa(double kappa) { kappa_param_.set_value(kappa); }
  void set_t0(double t0) { t0_param_.set_value(t0); }
  void set_init_buffer(unsigned int init_buffer) { init_buffer_param_.set_value(init_buffer); }
  void set_term_buffer(unsigned int term_buffer) { term_buffer_param_.set_value(term_buffer); }
  void set_window(unsigned int window) { window_param_.set_value(window); }
  void set_adaptation_engaged(bool adaptation_engaged) { adaptation_engaged_param_.set_value(adaptation_engaged); }

 private:
  metric_t metric_type_;
  std::shared_ptr<const stan::io::var_context> init_inv_metric_;
  
  // Parameter objects that encapsulate default values, validation, and descriptions
  stan::run::init_radius init_radius_param_;
  stan::run::stepsize stepsize_param_;
  stan::run::stepsize_jitter stepsize_jitter_param_;
  stan::run::max_depth max_depth_param_;
  stan::run::delta delta_param_;
  stan::run::gamma gamma_param_;
  stan::run::kappa kappa_param_;
  stan::run::t0 t0_param_;
  stan::run::init_buffer init_buffer_param_;
  stan::run::term_buffer term_buffer_param_;
  stan::run::window window_param_;
  stan::run::adaptation_engaged adaptation_engaged_param_;
};

}  // namespace run
}  // namespace stan
#endif
