#ifndef STAN_MCMC_HMC_XHMC_ADAPT_DENSE_E_XHMC_HPP
#define STAN_MCMC_HMC_XHMC_ADAPT_DENSE_E_XHMC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/stepsize_covar_adapter.hpp>
#include <stan/mcmc/hmc/xhmc/dense_e_xhmc.hpp>

namespace stan {
namespace mcmc {
/**
 * Exhaustive Hamiltonian Monte Carlo (XHMC) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and adaptive
 * dense metric and adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_dense_e_xhmc : public dense_e_xhmc<Model, BaseRNG>,
                           public stepsize_covar_adapter {
 public:
  adapt_dense_e_xhmc(const Model& model, BaseRNG& rng)
      : dense_e_xhmc<Model, BaseRNG>(model, rng),
        stepsize_covar_adapter(model.num_params_r()) {}

  ~adapt_dense_e_xhmc() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s = dense_e_xhmc<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                s.accept_stat());

      Eigen::MatrixXd inv_metric;

      bool update
          = this->covar_adaptation_.learn_covariance(inv_metric, this->z_.q);

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
