#ifndef STAN_MCMC_HMC_NUTS_DIAG_E_RB_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_DIAG_E_RB_NUTS_HPP

#include <stan/mcmc/hmc/rb_nuts/base_rb_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
  namespace mcmc {
    /**
     * The Rao-Blackwellized No-U-Turn sampler (NUTS) with multinomial weights
     * with a Gaussian-Euclidean disintegration and diagonal metric
     */
    template <class Model, class BaseRNG>
    class diag_e_rb_nuts : public base_rb_nuts<Model, diag_e_metric,
                                               expl_leapfrog, BaseRNG> {
    public:
      diag_e_rb_nuts(const Model& model, BaseRNG& rng)
        : base_rb_nuts<Model, diag_e_metric, expl_leapfrog,
                       BaseRNG>(model, rng) { }
    };

  }  // mcmc
}  // stan
#endif
