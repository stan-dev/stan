#ifndef STAN_RUN_NUTS_ADAPT_CONFIG_HPP
#define STAN_RUN_NUTS_ADAPT_CONFIG_HPP

#include <stan/io/var_context.hpp>
#include <stan/run/nuts_adapt_defaults.hpp>
#include <memory>

namespace stan {
namespace run {

class nuts_adapt_config {
public:
  // nuts_adapt_config_builder class embedded as friend
  class nuts_adapt_config_builder {
    friend class nuts_adapt_config;
    stan::run::delta delta_;
    stan::run::gamma gamma_;
    stan::run::kappa kappa_;
    stan::run::t0 t0_;
    stan::run::init_buffer init_buffer_;
    stan::run::term_buffer term_buffer_;
    stan::run::window window_;

  public:
    nuts_adapt_config_builder() : 
      delta_(),
      gamma_(),
      kappa_(),
      t0_(),
      init_buffer_(),
      term_buffer_(),
      window_() {}

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
  double delta() const { return delta_.value(); }
  double gamma() const { return gamma_.value(); }
  double kappa() const { return kappa_.value(); }
  double t0() const { return t0_.value(); }
  unsigned int init_buffer() const { return init_buffer_.value(); }
  unsigned int term_buffer() const { return term_buffer_.value(); }
  unsigned int window() const { return window_.value(); }

private:
  explicit nuts_adapt_config(const nuts_adapt_config_builder& builder) : 
    delta_(builder.delta_),
    gamma_(builder.gamma_),
    kappa_(builder.kappa_),
    t0_(builder.t0_),
    init_buffer_(builder.init_buffer_),
    term_buffer_(builder.term_buffer_),
    window_(builder.window_) {
  }

  stan::run::delta delta_;
  stan::run::gamma gamma_;
  stan::run::kappa kappa_;
  stan::run::t0 t0_;
  stan::run::init_buffer init_buffer_;
  stan::run::term_buffer term_buffer_;
  stan::run::window window_;
};

}  // namespace run
}  // namespace stan
#endif
