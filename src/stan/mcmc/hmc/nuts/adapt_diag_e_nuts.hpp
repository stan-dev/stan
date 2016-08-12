#ifndef STAN_MCMC_HMC_NUTS_ADAPT_DIAG_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_ADAPT_DIAG_E_NUTS_HPP

#include <stan/callbacks/writer.hpp>
#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/nuts/diag_e_nuts.hpp>

namespace stan {
  namespace mcmc {
    /**
     * The No-U-Turn sampler (NUTS) with multinomial sampling
     * with a Gaussian-Euclidean disintegration and adaptive
     * diagonal metric and adaptive step size
     */
    template <class Model, class BaseRNG>
    class adapt_diag_e_nuts: public diag_e_nuts<Model, BaseRNG>,
                             public stepsize_var_adapter {
    public:
        adapt_diag_e_nuts(const Model& model, BaseRNG& rng)
          : diag_e_nuts<Model, BaseRNG>(model, rng),
          stepsize_var_adapter(model.num_params_r()) {}

      ~adapt_diag_e_nuts() {}

      sample
      transition(sample& init_sample,
                 callbacks::writer& info_writer,
                 callbacks::writer& error_writer) {
        sample s = diag_e_nuts<Model, BaseRNG>::transition(init_sample,
                                                           info_writer,
                                                           error_writer);

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());

          bool update = this->var_adaptation_.learn_variance(this->z_.mInv,
                                                             this->z_.q);

          if (update) {
            this->init_stepsize(info_writer, error_writer);

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

  }  // mcmc
}  // stan
#endif
