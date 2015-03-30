#ifndef STAN__MCMC__ADAPT__UNIT__E__NUTS__BETA
#define STAN__MCMC__ADAPT__UNIT__E__NUTS__BETA

#include <stan/mcmc/stepsize_adapter.hpp>
#include <stan/mcmc/hmc/nuts/unit_e_nuts.hpp>

namespace stan {

  namespace mcmc {

    // The No-U-Turn Sampler (NUTS) on a
    // Euclidean manifold with unit metric
    // and adaptive stepsize

    template <class M, class BaseRNG, class Writer>
    class adapt_unit_e_nuts: public unit_e_nuts<M, BaseRNG, Writer>,
                             public stepsize_adapter {
    public:
      adapt_unit_e_nuts(M &m, BaseRNG& rng, Writer& writer)
        : unit_e_nuts<M, BaseRNG, Writer>(m, rng, writer) {}

      ~adapt_unit_e_nuts() {}

      sample transition(sample& init_sample) {
        sample s = unit_e_nuts<M, BaseRNG, Writer>::transition(init_sample);

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
