#ifndef STAN_MCMC_HMC_EXHAUSTIVE_ADAPT_UNIT_E_EXHAUSTIVE_HPP
#define STAN_MCMC_HMC_EXHAUSTIVE_ADAPT_UNIT_E_EXHAUSTIVE_HPP

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/exhaustive/unit_e_exhaustive.hpp>

namespace stan {
  namespace mcmc {

    // Exhaustive Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric
    // and adaptive stepsize

    template <typename M, class BaseRNG>
    class adapt_unit_e_exhaustive: public unit_e_exhaustive<M, BaseRNG>,
                             public stepsize_adapter {
    public:
      adapt_unit_e_exhaustive(M &m, BaseRNG& rng,
                        std::ostream* o, std::ostream* e)
        : unit_e_exhaustive<M, BaseRNG>(m, rng, o, e) {}

      ~adapt_unit_e_exhaustive() {}

      sample transition(sample& init_sample) {
        sample s = unit_e_exhaustive<M, BaseRNG>::transition(init_sample);

        if (this->adapt_flag_)
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());

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
