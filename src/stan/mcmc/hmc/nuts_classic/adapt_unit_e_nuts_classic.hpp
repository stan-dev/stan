#ifndef STAN_MCMC_HMC_NUTS_CLASSIC_ADAPT_UNIT_E_NUTS_CLASSIC_HPP
#define STAN_MCMC_HMC_NUTS_CLASSIC_ADAPT_UNIT_E_NUTS_CLASSIC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/nuts_classic/unit_e_nuts_classic.hpp>
#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
namespace mcmc {

// The No-U-Turn Sampler (NUTS) on a
// Euclidean manifold with unit metric
// and adaptive stepsize

template <class Model, class BaseRNG>
class adapt_unit_e_nuts_classic : public unit_e_nuts_classic<Model, BaseRNG>,
                                  public stepsize_adapter {
 public:
  adapt_unit_e_nuts_classic(const Model& model, BaseRNG& rng)
      : unit_e_nuts_classic<Model, BaseRNG>(model, rng) {}

  inline sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s
        = unit_e_nuts_classic<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_)
      this->nom_epsilon_
          = this->stepsize_adaptation_.learn_stepsize(s.accept_stat());

    return s;
  }

  inline void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    this->nom_epsilon_ = this->stepsize_adaptation_.complete_adaptation();
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
