#ifndef STAN_MCMC_HMC_STATIC_ADAPT_DENSE_E_STATIC_HMC_HPP
#define STAN_MCMC_HMC_STATIC_ADAPT_DENSE_E_STATIC_HMC_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/stepsize_covar_adapter.hpp>
#include <stan/mcmc/hmc/static/dense_e_static_hmc.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with dense metric,
    // static integration time,
    // and adaptive stepsize
    template <typename Model, class BaseRNG>
    class adapt_dense_e_static_hmc : public dense_e_static_hmc<Model, BaseRNG>,
                                     public stepsize_covar_adapter {
    public:
      adapt_dense_e_static_hmc(Model &model, BaseRNG& rng)
        : dense_e_static_hmc<Model, BaseRNG>(model, rng),
        stepsize_covar_adapter(model.num_params_r()) { }

      ~adapt_dense_e_static_hmc() { }

      sample transition(sample& init_sample,
                        interface_callbacks::writer::base_writer& writer) {
        sample s = dense_e_static_hmc<Model, BaseRNG>::transition(init_sample,
                                                                  writer);

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());
          this->update_L_();

          bool update = this->covar_adaptation_.learn_covariance
            (this->z_.mInv, this->z_.q);

          if (update) {
            this->init_stepsize(writer);
            this->update_L_();

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
