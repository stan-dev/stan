#ifndef STAN_MCMC_HMC_STATIC_UNIFORM_ADAPT_UNIT_E_STATIC_UNIFORM_HPP
#define STAN_MCMC_HMC_STATIC_UNIFORM_ADAPT_UNIT_E_STATIC_UNIFORM_HPP

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/static_uniform/unit_e_static_uniform.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric,
    // static integration time,
    // and adaptive stepsize
    template <typename M, class BaseRNG>
    class adapt_unit_e_static_uniform : public unit_e_static_uniform<M, BaseRNG>,
                                    public stepsize_adapter {
    public:
      adapt_unit_e_static_uniform(M &m, BaseRNG& rng,
                              std::ostream* o, std::ostream* e)
        : unit_e_static_uniform<M, BaseRNG>(m, rng, o, e) { }

      ~adapt_unit_e_static_uniform() { }

      sample transition(sample& init_sample) {
        sample s = unit_e_static_uniform<M, BaseRNG>::transition(init_sample);

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());
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
