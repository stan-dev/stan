#ifndef STAN_MCMC_HMC_BASE_HMC_HPP
#define STAN_MCMC_HMC_BASE_HMC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <boost/random/uniform_01.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {
namespace mcmc {

/**
 * Base class for Hamiltonian samplers.
 *
 * @tparam Model The type of the Stan model.
 * @tparam Hamiltonian The type of Hamiltonians over the (unconstrained)
 * parameter space.
 * @tparam Integrator The type of integrator (e.g. leapfrog).
 * @tparam BaseRNG The type of random number generator.
 */
template <class Model, template <class, class> class Hamiltonian,
          template <class> class Integrator, class BaseRNG>
class base_hmc : public base_mcmc {
 public:
  base_hmc(const Model& model, BaseRNG& rng)
      : base_mcmc(),
        z_(model.num_params_r()),
        integrator_(),
        hamiltonian_(model),
        rand_int_(rng),
        rand_uniform_(rand_int_),
        nom_epsilon_(0.1),
        epsilon_(nom_epsilon_),
        epsilon_jitter_(0.0) {}

  /**
   * format and write stepsize
   */
  void write_sampler_stepsize(callbacks::writer& writer) {
    std::stringstream nominal_stepsize;
    nominal_stepsize << "Step size = " << get_nominal_stepsize();
    writer(nominal_stepsize.str());
  }

  /**
   * write elements of mass matrix
   */
  void write_sampler_metric(callbacks::writer& writer) {
    z_.write_metric(writer);
  }

  /**
   * write stepsize and elements of mass matrix
   */
  void write_sampler_state(callbacks::writer& writer) {
    write_sampler_stepsize(writer);
    write_sampler_metric(writer);
  }

  void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                    std::vector<std::string>& names) {
    z_.get_param_names(model_names, names);
  }

  void get_sampler_diagnostics(std::vector<double>& values) {
    z_.get_params(values);
  }

  void seed(const Eigen::VectorXd& q) { z_.q = q; }

  void init_hamiltonian(callbacks::logger& logger) {
    this->hamiltonian_.init(this->z_, logger);
  }

  void init_stepsize(callbacks::logger& logger) {
    ps_point z_init(this->z_);

    // Skip initialization for extreme step sizes
    if (this->nom_epsilon_ == 0 || this->nom_epsilon_ > 1e7
        || std::isnan(this->nom_epsilon_))
      return;

    this->hamiltonian_.sample_p(this->z_, this->rand_int_);
    this->hamiltonian_.init(this->z_, logger);

    // Guaranteed to be finite if randomly initialized
    double H0 = this->hamiltonian_.H(this->z_);

    this->integrator_.evolve(this->z_, this->hamiltonian_, this->nom_epsilon_,
                             logger);

    double h = this->hamiltonian_.H(this->z_);
    if (std::isnan(h))
      h = std::numeric_limits<double>::infinity();

    double delta_H = H0 - h;

    int direction = delta_H > std::log(0.8) ? 1 : -1;

    while (1) {
      this->z_.ps_point::operator=(z_init);

      this->hamiltonian_.sample_p(this->z_, this->rand_int_);
      this->hamiltonian_.init(this->z_, logger);

      double H0 = this->hamiltonian_.H(this->z_);

      this->integrator_.evolve(this->z_, this->hamiltonian_, this->nom_epsilon_,
                               logger);

      double h = this->hamiltonian_.H(this->z_);
      if (std::isnan(h))
        h = std::numeric_limits<double>::infinity();

      double delta_H = H0 - h;

      if ((direction == 1) && !(delta_H > std::log(0.8)))
        break;
      else if ((direction == -1) && !(delta_H < std::log(0.8)))
        break;
      else
        this->nom_epsilon_ = direction == 1 ? 2.0 * this->nom_epsilon_
                                            : 0.5 * this->nom_epsilon_;

      if (this->nom_epsilon_ > 1e7)
        throw std::runtime_error(
            "Posterior is improper. "
            "Please check your model.");
      if (this->nom_epsilon_ == 0)
        throw std::runtime_error(
            "No acceptably small step size could "
            "be found. Perhaps the posterior is "
            "not continuous?");
    }

    this->z_.ps_point::operator=(z_init);
  }

  /**
   * Gets the current point in the (unconstrained) parameter space.
   *
   * @return The current point in the (unconstrained) parameter space.
   */
  typename Hamiltonian<Model, BaseRNG>::PointType& z() { return z_; }

  /**
   * Gets the current point in the (unconstrained) parameters space.
   *
   * @return The current point in the (unconstrained) parameters space.
   */
  const typename Hamiltonian<Model, BaseRNG>::PointType& z() const noexcept {
    return z_;
  }

  virtual void set_nominal_stepsize(double e) {
    if (e > 0)
      nom_epsilon_ = e;
  }

  double get_nominal_stepsize() { return this->nom_epsilon_; }

  double get_current_stepsize() { return this->epsilon_; }

  virtual void set_stepsize_jitter(double j) {
    if (j > 0 && j < 1)
      epsilon_jitter_ = j;
  }

  double get_stepsize_jitter() { return this->epsilon_jitter_; }

  void sample_stepsize() {
    this->epsilon_ = this->nom_epsilon_;
    if (this->epsilon_jitter_)
      this->epsilon_
          *= 1.0 + this->epsilon_jitter_ * (2.0 * this->rand_uniform_() - 1.0);
  }

 protected:
  typename Hamiltonian<Model, BaseRNG>::PointType z_;
  Integrator<Hamiltonian<Model, BaseRNG> > integrator_;
  Hamiltonian<Model, BaseRNG> hamiltonian_;

  BaseRNG& rand_int_;

  // Uniform(0, 1) RNG
  boost::uniform_01<BaseRNG&> rand_uniform_;

  double nom_epsilon_;
  double epsilon_;
  double epsilon_jitter_;
};

}  // namespace mcmc
}  // namespace stan
#endif
