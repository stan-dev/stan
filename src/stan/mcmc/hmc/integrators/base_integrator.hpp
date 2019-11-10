#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <stan/callbacks/logger.hpp>

namespace stan {
namespace mcmc {

template <class Hamiltonian>
class base_integrator {
 public:
  base_integrator() {}

  virtual void evolve(typename Hamiltonian::point_type& z,
                      Hamiltonian& hamiltonian, const double epsilon,
                      callbacks::logger& logger)
      = 0;
};

}  // namespace mcmc
}  // namespace stan
#endif
