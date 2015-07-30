#ifndef STAN_MCMC_HMC_EXHAUSTIVE_DIAG_E_EXHAUSTIVE_HPP
#define STAN_MCMC_HMC_EXHAUSTIVE_DIAG_E_EXHAUSTIVE_HPP

#include <stan/mcmc/hmc/exhaustive/base_exhaustive.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {
    // Exhaustive Hamiltonian Monte Carlo on a
    // Euclidean manifold with diagonal metric

    template <typename M, class BaseRNG>
    class diag_e_exhaustive : public base_exhaustive<M, diag_e_point, diag_e_metric,
                                         expl_leapfrog, BaseRNG> {
    public:
      diag_e_exhaustive(M &m, BaseRNG& rng, std::ostream* o,
                  std::ostream* e)
        : base_exhaustive<M, diag_e_point, diag_e_metric, expl_leapfrog,
                    BaseRNG>(m, rng, o, e) {
        this->name_ = "Exhaustive HMC with a diagonal Euclidean metric";
      }
                                           
    };
  }  // mcmc
}  // stan

#endif
