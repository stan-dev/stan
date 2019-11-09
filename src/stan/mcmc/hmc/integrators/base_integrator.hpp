#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <stan/callbacks/logger.hpp>

namespace stan {
namespace mcmc {

template <typename Derived, typename Hamiltonian>
class base_integrator {
 public:
  base_integrator() {}
  using point_type = typename Hamiltonian::point_type;
  using hamiltonian_type = Hamiltonian;
  Derived& derived() { return static_cast<Derived&>(*this); }
  const Derived& derived() const { return static_cast<Derived const&>(*this); }

  inline void evolve(point_type& z, hamiltonian_type& hamiltonian,
                     const double epsilon, callbacks::logger& logger) {
    this->derived().evolve(z, epsilon, logger);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
