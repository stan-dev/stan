#ifndef STAN_RUN_NUTS_ADAPT_CONFIG_HPP
#define STAN_RUN_NUTS_ADAPT_CONFIG_HPP

#include <stan/io/var_context.hpp>
#include <stan/run/config/nuts_adapt_defaults.hpp>
#include <memory>

class nuts_adapt_config {
 public:
  enum class metric_t {
    UNIT_E,
    DIAG_E,
    DENSE_E
  };

  // nuts_adapt_config_builder class embedded as friend
  class nuts_adapt_config_builder {
    friend class nuts_adapt_config;
    metric_t metric_type_ = metric_t::DIAG_E;
    std::shared_ptr<const stan::io::var_context> init_inv_metric_ = nullptr;
    double stepsize_ = stepsize::default_value();
    double stepsize_jitter_ = stepsize_jitter::default_value();
    int max_depth_ = max_depth::default_value();
    double delta_ = delta::default_value();
    double gamma_ = gamma::default_value();
    double kappa_ = kappa::default_value();
    double t0_ = t0::default_value();
    unsigned int init_buffer_ = init_buffer::default_value();
    unsigned int term_buffer_ = term_buffer::default_value();
    unsigned int window_ = window::default_value();

   public:
    nuts_adapt_config_builder& metric_type(metric_t metric_type) {
      metric_type_ = metric_type;
      return *this;
    }
    nuts_adapt_config_builder&
    init_inv_metric(std::shared_ptr<const stan::io::var_context> inv_metric) {
      init_inv_metric_ = inv_metric;
      return *this;
    }
    nuts_adapt_config_builder& stepsize(double size) {
      stepsize_ = stan::run::stepsize(size);
      return *this;
    }
    nuts_adapt_config_builder& stepsize_jitter(double jitter) {
      stepsize_jitter_ = stan::run::stepsize_jitter(jitter);
      return *this;
    }
    nuts_adapt_config_builder& max_depth(int depth) {
      max_depth_ = stan::run::depth(depth);
      return *this;
    }
    nuts_adapt_config_builder& delta(double d) {
      delta_ = stan::run::delta(d);
      return *this;
    }
    nuts_adapt_config_builder& gamma(double g) {
      gamma_ = stan::run::gamma(g);
      return *this;
    }
    nuts_adapt_config_builder& kappa(double k) {
      kappa_ = stan::run::kappa(k);
      return *this;
    }
    nuts_adapt_config_builder& t0(double t) {
      t0_ = stan::run::t0(t);
      return *this;
    }
    nuts_adapt_config_builder& init_buffer(unsigned int buffer) {
      init_buffer_ = stan::run::init_buffer(buffer);
      return *this;
    }
    nuts_adapt_config_builder& term_buffer(unsigned int buffer) {
      term_buffer_ = stan::run::term_buffer(buffer);
      return *this;
    }
    nuts_adapt_config_builder& window(unsigned int w) {
      window_ = stan::run::window(w);
      return *this;
    }

    nuts_adapt_config build() {
      validate();
      return nuts_adapt_config(*this);
    }

    void validate() const {
      // check inconsistencies between args
      // would need to check hmc_schedule vs. mcmc_iter schedule
    }
  };

  static nuts_adapt_config_builder create() {
    return nuts_adapt_config_builder();
  }

  // Getters
  metric_t metric_type() const { return metric_type_; }
  std::shared_ptr<const stan::io::var_context> init_inv_metric() const {
    return init_inv_metric_;
  }
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

 private:
  explicit
  nuts_adapt_config(const nuts_adapt_config_builder& nuts_adapt_config_builder) :
    delta_param_(nuts_adapt_config_builder.delta_),
    gamma_param_(nuts_adapt_config_builder.gamma_),
    kappa_param_(nuts_adapt_config_builder.kappa_),
    t0_param_(nuts_adapt_config_builder.t0_),
    init_buffer_param_(nuts_adapt_config_builder.init_buffer_),
    term_buffer_param_(nuts_adapt_config_builder.term_buffer_),
    window_param_(nuts_adapt_config_builder.window_)
  }

  stan::run::delta delta_param_;
  stan::run::gamma gamma_param_;
  stan::run::kappa kappa_param_;
  stan::run::t0 t0_param_;
  stan::run::init_buffer init_buffer_param_;
  stan::run::term_buffer term_buffer_param_;
  stan::run::window window_param_;
};

}  // namespace run
}  // namespace stan
#endif
