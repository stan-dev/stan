#ifndef STAN_MCMC_HMC_STATIC_ADAPT_UNIT_E_STATIC_HMC_HPP
#define STAN_MCMC_HMC_STATIC_ADAPT_UNIT_E_STATIC_HMC_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/hmc/static/unit_e_static_hmc.hpp>
#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
  namespace mcmc {
    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric,
    // static integration time,
    // and adaptive stepsize
    template <class Model, class BaseRNG>
    class adapt_unit_e_static_uniform:
      public unit_e_static_uniform<Model, BaseRNG>,
      public stepsize_adapter {
    public:
      adapt_unit_e_static_uniform(Model &model, BaseRNG& rng):
        unit_e_static_uniform<Model, BaseRNG>(model, rng) { }

      ~adapt_unit_e_static_uniform() { }

      sample transition(sample& init_sample,
                        interface_callbacks::writer::base_writer& writer) {
        sample s
          = unit_e_static_uniform<Model, BaseRNG>::transition(init_sample,
                                                              writer);

        if (this->adapt_flag_) {
          this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                    s.accept_stat());
          this->update_L_();
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
