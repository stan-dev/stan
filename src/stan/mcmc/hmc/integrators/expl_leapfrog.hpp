#ifndef STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
namespace mcmc {

template <class Hamiltonian>
class expl_leapfrog : public base_leapfrog<Hamiltonian> {
 public:
  expl_leapfrog() : base_leapfrog<Hamiltonian>() {}
  using point_type = typename Hamiltonian::PointType;

  inline void begin_update_p(point_type& z,
                      Hamiltonian& hamiltonian, double epsilon,
                      callbacks::logger& logger) {
    z.p -= epsilon * hamiltonian.dphi_dq(z, logger);
  }

  inline void update_q(point_type& z, Hamiltonian& hamiltonian,
                double epsilon, callbacks::logger& logger) {
    z.q += epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);
  }

  inline void end_update_p(point_type& z,
                    Hamiltonian& hamiltonian, double epsilon,
                    callbacks::logger& logger) {
    z.p -= epsilon * hamiltonian.dphi_dq(z, logger);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
