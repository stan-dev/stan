#ifndef STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  namespace mcmc {

    template <class Hamiltonian>
    class expl_leapfrog : public base_leapfrog<Hamiltonian> {
    public:
      expl_leapfrog()
        : base_leapfrog<Hamiltonian>() {}

      void begin_update_p(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian, double epsilon,
                          interface_callbacks::writer::base_writer& writer) {
        z.p -= epsilon * hamiltonian.dphi_dq(z, writer);
      }

      void update_q(typename Hamiltonian::PointType& z,
                    Hamiltonian& hamiltonian, double epsilon,
                    interface_callbacks::writer::base_writer& writer) {
        z.q += epsilon * hamiltonian.dtau_dp(z);
        hamiltonian.update_potential_gradient(z, writer);
      }

      void end_update_p(typename Hamiltonian::PointType& z,
                        Hamiltonian& hamiltonian, double epsilon,
                        interface_callbacks::writer::base_writer& writer) {
        z.p -= epsilon * hamiltonian.dphi_dq(z, writer);
      }
    };

  }  // mcmc
}  // stan
#endif
