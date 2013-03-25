#ifndef __STAN__MCMC__INTEGRATOR__BETA__
#define __STAN__MCMC__INTEGRATOR__BETA__

#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/hamiltonian.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    template <typename H>
    class integrator {
      
    public:
      
      virtual void evolve(psPoint& z, H hamiltonian, const double epsilon) = 0;
      
    };
    
    template <typename H>
    class leapfrog: public integrator<H> {
      
    public:
      
      void evolve(psPoint& z, H hamiltonian, const double epsilon) {
        beginUpdateP(z, hamiltonian, 0.5 * epsilon);
        updateQ(z, hamiltonian, epsilon);
        endUpdateP(z, hamiltonian, 0.5 * epsilon);
      }
      
      virtual void beginUpdateP(psPoint& z, H hamiltonian, double epsilon) = 0;
      virtual void updateQ(psPoint& z, H hamiltonian, double epsilon) = 0;
      virtual void endUpdateP(psPoint& z, H hamiltonian, double epsilon) = 0;
      
    };
    
    template <typename H>
    class expl_leapfrog: public leapfrog<H> {
      
    public:
      
      void beginUpdateP(psPoint& z, H hamiltonian, double epsilon) { 
        z.p += epsilon * hamiltonian.dphi_dq(z); 
      }
      
      void updateQ(psPoint& z, H hamiltonian, double epsilon) { 
        Eigen::Map<Eigen::VectorXd> q(&(z.q[0]), z.q.size());
        q += epsilon * hamiltonian.dtau_dp(z); 
      }
      
      void endUpdateP(psPoint& z, H hamiltonian, double epsilon) { 
        z.p += epsilon * hamiltonian.dphi_dq(z); 
      }
      
    };
    
    // implicit leapfrog
    
  } // mcmc

} // stan
          

#endif
