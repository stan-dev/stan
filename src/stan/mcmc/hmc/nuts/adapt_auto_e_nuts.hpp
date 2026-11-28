#ifndef STAN_MCMC_HMC_NUTS_ADAPT_AUTO_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_ADAPT_AUTO_E_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/stepsize_auto_adapter.hpp>
#include <stan/mcmc/hmc/nuts/auto_e_nuts.hpp>

namespace stan {
namespace mcmc {
/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and adaptive
 * dense metric and adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_auto_e_nuts : public auto_e_nuts<Model, BaseRNG>,
                          public stepsize_auto_adapter {
 protected:
  const Model& model_;

 public:
  adapt_auto_e_nuts(const Model& model, BaseRNG& rng)
      : model_(model),
        auto_e_nuts<Model, BaseRNG>(model, rng),
        stepsize_auto_adapter(model.num_params_r()) {}

  ~adapt_auto_e_nuts() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s = auto_e_nuts<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                s.accept_stat());

      bool update = this->auto_adaptation_.learn_covariance(
          model_, this->z_.inv_e_metric_, this->z_.is_diagonal_, this->z_.q);

      if (update) {
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
