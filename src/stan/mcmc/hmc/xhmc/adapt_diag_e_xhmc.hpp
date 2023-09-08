#ifndef STAN_MCMC_HMC_XHMC_ADAPT_DIAG_E_XHMC_HPP
#define STAN_MCMC_HMC_XHMC_ADAPT_DIAG_E_XHMC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/xhmc/diag_e_xhmc.hpp>

namespace stan {
namespace mcmc {
/**
 * Exhaustive Hamiltonian Monte Carlo (XHMC) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and adaptive
 * diagonal metric and adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_diag_e_xhmc : public diag_e_xhmc<Model, BaseRNG>,
                          public stepsize_var_adapter {
 public:
  adapt_diag_e_xhmc(const Model& model, BaseRNG& rng)
      : diag_e_xhmc<Model, BaseRNG>(model, rng),
        stepsize_var_adapter(model.num_params_r()) {}

  ~adapt_diag_e_xhmc() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s = diag_e_xhmc<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                s.accept_stat());

      Eigen::VectorXd inv_metric;

      bool update
          = this->var_adaptation_.learn_variance(inv_metric, this->z_.q);

      if (update) {
        this->z_.set_inv_metric(std::move(inv_metric));

        this->init_stepsize(logger);

        this->stepsize_adaptation_.set_mu(log(10 * this->nom_epsilon_));
        this->stepsize_adaptation_.restart();
      }
    }
    return s;
  }

  void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    this->stepsize_adaptation_.complete_adaptation(this->nom_epsilon_);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
