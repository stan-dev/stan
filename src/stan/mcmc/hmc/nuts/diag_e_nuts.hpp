#ifndef STAN_MCMC_HMC_NUTS_DIAG_E_NUTS_HPP
#define STAN_MCMC_HMC_NUTS_DIAG_E_NUTS_HPP

#include <stan/mcmc/hmc/nuts/base_nuts.hpp>
#include <stan/mcmc/hmc/nuts/base_parallel_nuts.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
namespace mcmc {

/**
 * The No-U-Turn sampler (NUTS) with multinomial sampling
 * with a Gaussian-Euclidean disintegration and diagonal metric
 */
template <class Model, class BaseRNG, bool ParallelBase = false>
class diag_e_nuts : public base_nuts_ct<ParallelBase, Model, diag_e_metric,
                                        expl_leapfrog, BaseRNG> {
  using base_nuts_t = base_nuts_ct<ParallelBase, Model, diag_e_metric,
                                   expl_leapfrog, BaseRNG>;

 public:
  template <bool ParallelBase_ = ParallelBase,
            std::enable_if_t<!ParallelBase_>* = nullptr>
  diag_e_nuts(const Model& model, BaseRNG& rng) : base_nuts_t(model, rng) {}
  template <bool ParallelBase_ = ParallelBase,
            std::enable_if_t<ParallelBase_>* = nullptr>
  diag_e_nuts(const Model& model, std::vector<BaseRNG>& thread_rngs)
      : base_nuts_t(model, thread_rngs) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
