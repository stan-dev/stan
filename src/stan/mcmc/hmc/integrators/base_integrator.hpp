#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <ostream>

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class base_integrator {
    public:
      explicit base_integrator(std::ostream* o)
        : out_stream_(o) {}

      virtual void evolve(P& z, H& hamiltonian, const double epsilon) = 0;

    protected:
      std::ostream* out_stream_;
    };

  }  // mcmc

}  // stan

#endif
