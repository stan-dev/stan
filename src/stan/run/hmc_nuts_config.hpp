#ifndef STAN_RUN_HMC_NUTS_CONFIG_HPP
#define STAN_RUN_HMC_NUTS_CONFIG_HPP

#include <stan/run/process_config.hpp>
#include <stan/run/hmc_config.hpp>
#include <stan/run/nuts_adapt_config.hpp>
#include <stdexcept>
#include <sstream>

namespace stan {
namespace run {

/**
 * Configuration class for HMC NUTS sampling.
 * 
 * Composes process configuration (chains, initialization, output),
 * HMC parameters (sampling, stepsize, metric), NUTS adaptation
 * (target acceptance, adaptation schedule), and output options.
 */
class hmc_nuts_config {
public:
  /**
   * Builder class for constructing HMC NUTS configuration.
   * 
   * Provides fluent interface that forwards method calls to the
   * appropriate sub-builders and handles cross-component validation.
   */
  class hmc_nuts_config_builder {
    friend class hmc_nuts_config;
    num_chains_config num_chains_;
    process_config::process_config_builder process_builder_;
    hmc_config::hmc_config_builder hmc_builder_;
    nuts_adapt_config::nuts_adapt_config_builder adapt_builder_;
    
    // Output options
    bool save_start_params_ = false;
    bool save_warmup_ = false;
    bool save_diagnostics_ = false;
    bool save_metric_ = false;

  public:
    hmc_nuts_config_builder() : 
      process_builder_(), hmc_builder_(), adapt_builder_() {}

    hmc_nuts_config_builder& num_chains(size_t num_chains) {
      num_chains_ = num_chains_config(num_chains);
      return *this;
    }
    // Process configuration methods
    hmc_nuts_config_builder& num_chains(size_t num_chains) {
      process_builder_.num_chains(num_chains);
      return *this;
    }

    hmc_nuts_config_builder& output_dir(const std::string& output_dir) {
      process_builder_.output_dir(output_dir);
      return *this;
    }

    hmc_nuts_config_builder& seed(unsigned int seed) {
      process_builder_.seed(seed);
      return *this;
    }

    hmc_nuts_config_builder& init_radius(double radius) {
      process_builder_.init_radius(radius);
      return *this;
    }

    hmc_nuts_config_builder& init_params(const std::vector<std::string>& filenames) {
      process_builder_.init_params(filenames);
      return *this;
    }

    // HMC configuration methods
    hmc_nuts_config_builder& num_warmup(int warmup) {
      hmc_builder_.num_warmup(warmup);
      return *this;
    }

    hmc_nuts_config_builder& num_samples(int samples) {
      hmc_builder_.num_samples(samples);
      return *this;
    }

    hmc_nuts_config_builder& thin(int thin_value) {
      hmc_builder_.thin(thin_value);
      return *this;
    }

    hmc_nuts_config_builder& refresh(int refresh_value) {
      hmc_builder_.refresh(refresh_value);
      return *this;
    }

    hmc_nuts_config_builder& metric_type(metric_t metric_type) {
      hmc_builder_.metric_type(metric_type);
      return *this;
    }

    hmc_nuts_config_builder& init_inv_metric(
        std::shared_ptr<const stan::io::var_context> inv_metric) {
      hmc_builder_.init_inv_metric(inv_metric);
      return *this;
    }

    hmc_nuts_config_builder& stepsize(double size) {
      hmc_builder_.stepsize(size);
      return *this;
    }

    hmc_nuts_config_builder& stepsize_jitter(double jitter) {
      hmc_builder_.stepsize_jitter(jitter);
      return *this;
    }

    hmc_nuts_config_builder& max_depth(int depth) {
      hmc_builder_.max_depth(depth);
      return *this;
    }

    // NUTS adaptation configuration methods
    hmc_nuts_config_builder& delta(double d) {
      adapt_builder_.delta(d);
      return *this;
    }

    hmc_nuts_config_builder& gamma(double g) {
      adapt_builder_.gamma(g);
      return *this;
    }

    hmc_nuts_config_builder& kappa(double k) {
      adapt_builder_.kappa(k);
      return *this;
    }

    hmc_nuts_config_builder& t0(double t) {
      adapt_builder_.t0(t);
      return *this;
    }

    hmc_nuts_config_builder& init_buffer(unsigned int buffer) {
      adapt_builder_.init_buffer(buffer);
      return *this;
    }

    hmc_nuts_config_builder& term_buffer(unsigned int buffer) {
      adapt_builder_.term_buffer(buffer);
      return *this;
    }

    hmc_nuts_config_builder& window(unsigned int w) {
      adapt_builder_.window(w);
      return *this;
    }

    // Output option methods
    hmc_nuts_config_builder& save_start_params(bool save) {
      save_start_params_ = save;
      return *this;
    }

    hmc_nuts_config_builder& save_warmup(bool save) {
      save_warmup_ = save;
      return *this;
    }

    hmc_nuts_config_builder& save_diagnostics(bool save) {
      save_diagnostics_ = save;
      return *this;
    }

    hmc_nuts_config_builder& save_metric(bool save) {
      save_metric_ = save;
      return *this;
    }

    /**
     * Builds the composed configuration.
     * Validates individual components and cross-component constraints.
     */
    hmc_nuts_config build() {
      validate();
      return hmc_nuts_config(*this);
    }

    /**
     * Validates the composed configuration.
     * Checks individual component validity and cross-component constraints.
     */
    void validate() const {
      // Validate individual components
      process_builder_.validate();
      hmc_builder_.validate();
      adapt_builder_.validate();
      
      // Cross-component validation
      validate_adaptation_schedule();
    }

  private:
    /**
     * Validates that adaptation schedule is consistent with sampling schedule.
     */
    void validate_adaptation_schedule() const {
      // Build temporary configs to access values for validation
      auto temp_hmc = hmc_builder_;
      temp_hmc.validate();  // This might throw, but that's OK
      auto hmc_config_temp = hmc_config(temp_hmc);
      
      auto temp_adapt = adapt_builder_;
      auto adapt_config = nuts_adapt_config(temp_adapt);
      
      int total_warmup = hmc_config_temp.num_warmup();
      unsigned int init_buffer = adapt_config.init_buffer();
      unsigned int term_buffer = adapt_config.term_buffer();
      unsigned int window = adapt_config.window();
      
      // Check that adaptation schedule fits within warmup
      if (init_buffer + term_buffer + window > static_cast<unsigned int>(total_warmup)) {
        std::stringstream msg;
        msg << "Adaptation schedule exceeds warmup iterations. "
            << "init_buffer(" << init_buffer << ") + "
            << "term_buffer(" << term_buffer << ") + "
            << "window(" << window << ") = "
            << (init_buffer + term_buffer + window)
            << " > num_warmup(" << total_warmup << ")";
        throw std::invalid_argument(msg.str());
      }
    }
  };

  /**
   * Creates a new builder for HMC NUTS configuration.
   */
  static hmc_nuts_config_builder create() {
    return hmc_nuts_config_builder();
  }

  // Access to composed configurations
  const process_config& process() const { return process_; }
  const hmc_config& hmc() const { return hmc_; }
  const nuts_adapt_config& adaptation() const { return adaptation_; }

  // Convenience getters for commonly used values
  // Process-level
  size_t num_chains() const { return process_.num_chains(); }
  unsigned int seed() const { return process_.seed(); }
  const std::string& output_dir() const { return process_.output_dir(); }
  double init_radius() const { return process_.init_radius(); }
  bool has_init_params() const { return process_.has_init_params(); }

  // HMC-level  
  int num_warmup() const { return hmc_.num_warmup(); }
  int num_samples() const { return hmc_.num_samples(); }
  int thin() const { return hmc_.thin(); }
  double stepsize() const { return hmc_.stepsize(); }
  metric_t metric_type() const { return hmc_.metric_type(); }

  // NUTS adaptation
  double delta() const { return adaptation_.delta(); }
  double gamma() const { return adaptation_.gamma(); }
  unsigned int init_buffer() const { return adaptation_.init_buffer(); }

  // Output options
  bool save_start_params() const { return save_start_params_; }
  bool save_warmup() const { return save_warmup_; }
  bool save_diagnostics() const { return save_diagnostics_; }
  bool save_metric() const { return save_metric_; }

private:
  /**
   * Private constructor that builds from the builder.
   */
  explicit hmc_nuts_config(const hmc_nuts_config_builder& builder) :
      process_(builder.process_builder_.build()),
      hmc_(builder.hmc_builder_.build()),
      adaptation_(builder.adapt_builder_.build()),
      save_start_params_(builder.save_start_params_),
      save_warmup_(builder.save_warmup_),
      save_diagnostics_(builder.save_diagnostics_),
      save_metric_(builder.save_metric_) {}

  process_config process_;
  hmc_config hmc_;
  nuts_adapt_config adaptation_;
  
  // Output options
  bool save_start_params_;
  bool save_warmup_;
  bool save_diagnostics_;
  bool save_metric_;
};

}  // namespace run
}  // namespace stan

#endif
