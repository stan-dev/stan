#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>

namespace stan {
  namespace mcmc {

    template <class Hamiltonian>
    class base_integrator {
    public:
      base_integrator() {}

      virtual void
      evolve(typename Hamiltonian::PointType& z,
             Hamiltonian& hamiltonian,
             const double epsilon,
             interface_callbacks::writer::base_writer& info_writer,
             interface_callbacks::writer::base_writer& error_writer) = 0;
    };

  }  // mcmc
}  // stan
#endif
