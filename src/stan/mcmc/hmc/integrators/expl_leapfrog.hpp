#ifndef STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP

#include <Eigen/Dense>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  namespace mcmc {

    template <class Hamiltonian>
    class expl_leapfrog : public base_leapfrog<Hamiltonian> {
    public:
      expl_leapfrog()
        : base_leapfrog<Hamiltonian>() {}

      void begin_update_p(
        typename Hamiltonian::PointType& z,
        Hamiltonian& hamiltonian, double epsilon,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        z.p -= epsilon * hamiltonian.dphi_dq(z, info_writer, error_writer);
      }

      void update_q(typename Hamiltonian::PointType& z,
                    Hamiltonian& hamiltonian, double epsilon,
                    interface_callbacks::writer::base_writer& info_writer,
                    interface_callbacks::writer::base_writer& error_writer) {
        z.q += epsilon * hamiltonian.dtau_dp(z);
        hamiltonian.update_potential_gradient(z, info_writer, error_writer);
      }

      void end_update_p(
        typename Hamiltonian::PointType& z,
        Hamiltonian& hamiltonian, double epsilon,
        interface_callbacks::writer::base_writer& info_writer,
        interface_callbacks::writer::base_writer& error_writer) {
        z.p -= epsilon * hamiltonian.dphi_dq(z, info_writer, error_writer);
      }
    };

  }  // mcmc
}  // stan
#endif
