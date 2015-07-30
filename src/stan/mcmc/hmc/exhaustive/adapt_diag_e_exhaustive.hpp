#ifndef STAN_MCMC_HMC_EXHAUSTIVE_ADAPT_DIAG_E_EXHAUSTIVE_HPP
#define STAN_MCMC_HMC_EXHAUSTIVE_ADAPT_DIAG_E_EXHAUSTIVE_HPP

#include <stan/mcmc/stepsize_var_adapter.hpp>
#include <stan/mcmc/hmc/exhaustive/diag_e_exhaustive.hpp>

namespace stan {
  namespace mcmc {

    // Exhaustive Hamiltonian Monte Carlo on a
    // Euclidean manifold with diagonal metric
    // and adaptive stepsize

    template <typename M, class BaseRNG>
    class adapt_diag_e_exhaustive: public diag_e_exhaustive<M, BaseRNG>,
                             public stepsize_var_adapter {
    public:
        adapt_diag_e_exhaustive(M &m, BaseRNG& rng,
                          std::ostream* o, std::ostream* e)
          : diag_e_exhaustive<M, BaseRNG>(m, rng, o, e),
          stepsize_var_adapter(m.num_params_r()) {}

      ~adapt_diag_e_exhaustive() {}

      sample transition(sample& init_sample) {
        sample s = diag_e_exhaustive<M, BaseRNG>::transition(init_sample);

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());

          bool update = this->var_adaptation_.learn_variance(this->z_.mInv,
                                                             this->z_.q);

          if (update) {
            this->init_stepsize();

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
