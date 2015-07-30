#ifndef STAN_MCMC_HMC_EXHAUSTIVE_DENSE_E_EXHAUSTIVE_HPP
#define STAN_MCMC_HMC_EXHAUSTIVE_DENSE_E_EXHAUSTIVE_HPP

#include <stan/mcmc/hmc/exhaustive/base_exhaustive.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {
    // Exhaustive Hamiltonian Monte Carlo on a
    // Euclidean manifold with dense metric
    template <typename M, class BaseRNG>
    class dense_e_exhaustive : public base_exhaustive<M, dense_e_point, dense_e_metric,
                                          expl_leapfrog, BaseRNG> {
    public:
      dense_e_exhaustive(M &m, BaseRNG& rng, std::ostream* o,
                   std::ostream* e)
        : base_exhaustive<M, dense_e_point, dense_e_metric, expl_leapfrog,
                    BaseRNG>(m, rng, o, e) {
        this->name_ = "Exhaustive HMC with a dense Euclidean metric";
      }
    };
  }  // mcmc
}  // stan

#endif
