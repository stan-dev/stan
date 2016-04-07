#ifndef STAN_MCMC_HMC_STATIC_UNIFORM_UNIT_E_STATIC_UNIFORM_HPP
#define STAN_MCMC_HMC_STATIC_UNIFORM_UNIT_E_STATIC_UNIFORM_HPP

#include <stan/mcmc/hmc/static_uniform/base_static_uniform.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {
    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric
    // and static integration time

    template <typename M, class BaseRNG>
    class unit_e_static_uniform
      : public base_static_uniform<M, unit_e_metric, expl_leapfrog, BaseRNG> {
    public:
      unit_e_static_uniform(M &m, BaseRNG& rng):
        base_static_uniform<M, unit_e_metric, expl_leapfrog, BaseRNG>(m, rng) {
      }
    };
  }  // mcmc
}  // stan

#endif
