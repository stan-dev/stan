#ifndef STAN_MCMC_STEPSIZE_ADAPTATION_HPP
#define STAN_MCMC_STEPSIZE_ADAPTATION_HPP

#include <stan/mcmc/base_adaptation.hpp>
#include <cmath>

namespace stan {

namespace mcmc {

class stepsize_adaptation : public base_adaptation {
 public:
  stepsize_adaptation() {}

  inline void set_mu(double m) { mu_ = m; }

  inline void set_delta(double d) {
    if (d > 0 && d < 1)
      delta_ = d;
  }

  inline void set_gamma(double g) {
    if (g > 0)
      gamma_ = g;
  }

  inline void set_kappa(double k) {
    if (k > 0)
      kappa_ = k;
  }
  inline void set_t0(double t) {
    if (t > 0)
      t0_ = t;
  }

  inline double get_mu() { return mu_; }

  inline double get_delta() { return delta_; }

  inline double get_gamma() { return gamma_; }

  inline double get_kappa() { return kappa_; }

  inline double get_t0() { return t0_; }

  inline void restart() {
    counter_ = 0;
    s_bar_ = 0;
    x_bar_ = 0;
  }

  /**
   * Computes the stepsize given the adaption statistic.
   * @param adapt_stat The adaption statistic (HALP: What this?)
   * @return Epsilon
   */
  inline double learn_stepsize(double adapt_stat) {
    ++counter_;

    adapt_stat = adapt_stat > 1 ? 1 : adapt_stat;

    // Nesterov Dual-Averaging of log(epsilon)
    const double eta = 1.0 / (counter_ + t0_);

    s_bar_ = (1.0 - eta) * s_bar_ + eta * (delta_ - adapt_stat);

    const double x = mu_ - s_bar_ * std::sqrt(counter_) / gamma_;
    const double x_eta = std::pow(counter_, -kappa_);

    x_bar_ = (1.0 - x_eta) * x_bar_ + x_eta * x;

    return std::exp(x);
  }

  inline double complete_adaptation() { return std::exp(x_bar_); }

 protected:
  double counter_{0};  // Adaptation iteration
  double s_bar_{0};    // Moving average statistic
  double x_bar_{0};    // Moving average parameter
  double mu_{0.5};       // Asymptotic mean of parameter
  double delta_{0.5};    // Target value of statistic
  double gamma_{0.05};    // Adaptation scaling
  double kappa_{0.75};    // Adaptation shrinkage
  double t0_{10};       // Effective starting iteration
};

}  // namespace mcmc

}  // namespace stan

#endif
