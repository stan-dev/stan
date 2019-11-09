#ifndef STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP
#define STAN_MCMC_HMC_INTEGRATORS_BASE_INTEGRATOR_HPP

#include <stan/callbacks/logger.hpp>

namespace stan {
namespace mcmc {

/**
 * Base CRTP class for the symplectic integrators
 * @tparam Derived a class with an /c evolve() method defined.
 * @tparam Hamiltonian A class that represents a hamiltonian
 */
template <typename Derived, typename Hamiltonian>
class base_integrator {
 public:
  base_integrator() {}
  using point_type = typename Hamiltonian::point_type;
  using hamiltonian_type = Hamiltonian;
  // modifier to the derived class
  Derived& derived() { return static_cast<Derived&>(*this); }
  // inspector to the derived class
  const Derived& derived() const { return static_cast<Derived const&>(*this); }

  inline void evolve(point_type& z, hamiltonian_type& hamiltonian,
                     const double epsilon, callbacks::logger& logger) {
    this->derived().evolve(z, epsilon, logger);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
