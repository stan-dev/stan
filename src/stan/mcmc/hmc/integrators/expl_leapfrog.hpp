#ifndef STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  namespace mcmc {

    template <typename Hamiltonian>
    class expl_leapfrog : public base_leapfrog<Hamiltonian> {
    public:
      explicit expl_leapfrog(interface_callbacks::writer::base_writer& writer)
        : base_leapfrog<Hamiltonian>(writer) {}

      void begin_update_p(typename Hamiltonian::PointType& z,
                          Hamiltonian& hamiltonian,
                          double epsilon) {
        z.p -= epsilon * hamiltonian.dphi_dq(z);
      }

      void update_q(typename Hamiltonian::PointType& z,
                    Hamiltonian& hamiltonian,
                    double epsilon) {
        Eigen::Map<Eigen::VectorXd> q(&(z.q[0]), z.q.size());
        q += epsilon * hamiltonian.dtau_dp(z);
      }

      void end_update_p(typename Hamiltonian::PointType& z,
                        Hamiltonian& hamiltonian,
                        double epsilon) {
        z.p -= epsilon * hamiltonian.dphi_dq(z);
      }
    };

  }  // mcmc
}  // stan

#endif
