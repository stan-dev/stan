#ifndef STAN_MCMC_HMC_STATIC_UNIT_E_STATIC_HMC_HPP
#define STAN_MCMC_HMC_STATIC_UNIT_E_STATIC_HMC_HPP

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {

    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric
    // and static integration time
    template <typename Model, class BaseRNG>
    class unit_e_static_hmc
      : public base_static_hmc<Model, unit_e_metric,
                               expl_leapfrog, BaseRNG> {
    public:
      unit_e_static_hmc(Model &model, BaseRNG& rng)
        : base_static_hmc<Model, unit_e_metric,
                          expl_leapfrog, BaseRNG>(model, rng) {
        this->name_ = "Static HMC with a unit Euclidean metric";
      }
    };

  }  // mcmc
}  // stan
#endif
