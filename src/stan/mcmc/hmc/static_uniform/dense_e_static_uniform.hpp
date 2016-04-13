#ifndef STAN_MCMC_HMC_STATIC_UNIFORM_DENSE_E_STATIC_UNIFORM_HPP
#define STAN_MCMC_HMC_STATIC_UNIFORM_DENSE_E_STATIC_UNIFORM_HPP

#include <stan/mcmc/hmc/static_uniform/base_static_uniform.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {
    /**
     * Hamiltonian Monte Carlo implementation that uniformly samples
     * from trajectories with a static integration time with a
     * Gaussian-Euclidean disintegration and dense metric
     */
    template <typename M, class BaseRNG>
    class dense_e_static_uniform
      : public base_static_uniform<M, dense_e_metric, expl_leapfrog, BaseRNG> {
    public:
      dense_e_static_uniform(M &m, BaseRNG& rng):
        base_static_uniform<M, dense_e_metric, expl_leapfrog, BaseRNG>(m, rng) {
      }
    };
  }  // mcmc
}  // stan

#endif
