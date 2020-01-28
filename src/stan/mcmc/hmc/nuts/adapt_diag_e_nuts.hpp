#ifndef STAN_MCMC_HMC_NUTS_ADAPT_DIAG_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_ADAPT_DIAG_E_NUTS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

#ifdef MPI_ADAPTED_WARMUP
#include <stan/mcmc/mpi_cross_chain_adapter.hpp>
#endif

namespace stan {
namespace mcmc {
/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and adaptive
 * diagonal metric and adaptive step size
 */
template <class Model, class BaseRNG>
#ifdef MPI_ADAPTED_WARMUP
class adapt_diag_e_nuts : public diag_e_nuts<Model, BaseRNG>,
                          public stepsize_var_adapter,
                          public mpi_cross_chain_adapter {
#else
class adapt_diag_e_nuts : public diag_e_nuts<Model, BaseRNG>,
                          public stepsize_var_adapter {
#endif
 public:
  adapt_diag_e_nuts(const Model& model, BaseRNG& rng)
      : diag_e_nuts<Model, BaseRNG>(model, rng),
        stepsize_var_adapter(model.num_params_r()) {}

  ~adapt_diag_e_nuts() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s = diag_e_nuts<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                s.accept_stat());

      bool update = this->var_adaptation_.learn_variance(this->z_.inv_e_metric_,
                                                         this->z_.q);

      if (update) {
        this->init_stepsize(logger);

        this->stepsize_adaptation_.set_mu(log(10 * this->nom_epsilon_));
        this->stepsize_adaptation_.restart();
      }

#ifdef MPI_ADAPTED_WARMUP
      this -> add_cross_chain_sample(this->z_.q, s.log_prob());
      double stepsize = this -> get_nominal_stepsize();
      this -> cross_chain_adaptation(stepsize, this->z_.inv_e_metric_, logger);
      if (this -> is_cross_chain_adapted()) {
        this -> set_nominal_stepsize(stepsize);
      }
#endif
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
