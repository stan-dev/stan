#ifndef STAN_MCMC_HMC_NUTS_CLASSIC_UNIT_E_NUTS_CLASSIC_HPP
#define STAN_MCMC_HMC_NUTS_CLASSIC_UNIT_E_NUTS_CLASSIC_HPP

#include <stan/mcmc/hmc/nuts_classic/base_nuts_classic.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {
namespace mcmc {
// The No-U-Turn Sampler (NUTS) on a
// Euclidean manifold with unit metric
template <class Model, class BaseRNG>
class unit_e_nuts_classic
    : public base_nuts_classic<Model, unit_e_metric, expl_leapfrog, BaseRNG> {
 public:
  unit_e_nuts_classic(const Model& model, BaseRNG& rng)
      : base_nuts_classic<Model, unit_e_metric, expl_leapfrog, BaseRNG>(model,
                                                                        rng) {}

  using point_type = typename unit_e_metric<Model, BaseRNG>::point_type;

  bool compute_criterion(point_type& start, point_type& finish,
                         Eigen::VectorXd& rho) {
    return finish.p.dot(rho - finish.p) > 0 && start.p.dot(rho - start.p) > 0;
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
