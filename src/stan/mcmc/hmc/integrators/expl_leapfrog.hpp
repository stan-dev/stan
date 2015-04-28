#ifndef STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP
#define STAN_MCMC_HMC_INTEGRATORS_EXPL_LEAPFROG_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class expl_leapfrog : public base_leapfrog<H, P> {
    public:
      explicit expl_leapfrog(std::ostream* o = 0)
        : base_leapfrog<H, P>(o) {}

      void begin_update_p(P& z, H& hamiltonian, double epsilon) {
        z.p -= epsilon * hamiltonian.dphi_dq(z);
      }

      void update_q(P& z, H& hamiltonian, double epsilon) {
        Eigen::Map<Eigen::VectorXd> q(&(z.q[0]), z.q.size());
        q += epsilon * hamiltonian.dtau_dp(z);
      }

      void end_update_p(P& z, H& hamiltonian, double epsilon) {
        z.p -= epsilon * hamiltonian.dphi_dq(z);
      }
    };

  }  // mcmc

}  // stan

#endif
