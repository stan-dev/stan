#ifndef STAN_MCMC_HMC_STATIC_DENSE_E_STATIC_HMC_HPP
#define STAN_MCMC_HMC_STATIC_DENSE_E_STATIC_HMC_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/expl_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    // Hamiltonian Monte Carlo on a
    // Euclidean manifold with dense metric
    // and static integration time
    template <typename Model, class BaseRNG>
    class dense_e_static_hmc
      : public base_static_hmc<Model, dense_e_metric,
                               expl_leapfrog, BaseRNG> {
    public:
      dense_e_static_hmc(Model &model, BaseRNG& rng,
                         stan::interface_callbacks::writer::base_writer& writer)
        : base_static_hmc<Model, dense_e_metric,
                          expl_leapfrog, BaseRNG>(model, rng, writer) {
        this->name_ = "Static HMC with a dense Euclidean metric";
      }
    };

  }  // mcmc

}  // stan

#endif
