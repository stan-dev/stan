#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <ostream>

namespace stan {
  namespace mcmc {

    template <typename Hamiltonian>
    class base_integrator {
    public:
      base_integrator() {}
      virtual void evolve(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian,
                          const double epsilon) = 0;
    };

  }  // mcmc
}  // stan

#endif
