#ifndef STAN__MCMC__UNIT__E__STATIC__HMC__BETA
#define STAN__MCMC__UNIT__E__STATIC__HMC__BETA

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with unit metric
    // and static integration time

    template <class M, class BaseRNG, class Writer>
    class unit_e_static_hmc
      : public base_static_hmc<M, unit_e_metric, expl_leapfrog, BaseRNG, Writer> {
    public:
      unit_e_static_hmc(M &m, BaseRNG& rng, Writer& writer)
        : base_static_hmc<M, unit_e_metric, expl_leapfrog, BaseRNG, Writer>(m, rng, writer) {
        this->name_ = "Static HMC with a unit Euclidean metric";
      }
    };

  }  // mcmc

}  // stan

#endif
