#ifndef __STAN__MCMC__INTEGRATOR__BETA__
#define __STAN__MCMC__INTEGRATOR__BETA__

#include <stan/mcmc/hmc_base.hpp>
#include <stan/mcmc/hamiltonian.hpp>
#include <stan/mcmc/util.hpp>

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class integrator {
      
    public:
      
      virtual void evolve(P& z, H& hamiltonian, const double epsilon) = 0;
      
    };
    
    template <typename H, typename P>
    class leapfrog: public integrator<H, P> {
      
    public:
      
      void evolve(P& z, H& hamiltonian, const double epsilon) {

        begin_update_p(z, hamiltonian, 0.5 * epsilon);
        
        update_q(z, hamiltonian, epsilon);
        hamiltonian.update(z);
        
        end_update_p(z, hamiltonian, 0.5 * epsilon);
        
      }
      
      virtual void begin_update_p(P& z, H& hamiltonian, double epsilon) = 0;
      virtual void update_q(P& z, H& hamiltonian, double epsilon) = 0;
      virtual void end_update_p(P& z, H& hamiltonian, double epsilon) = 0;
      
    protected:
      
    };
    
    template <typename H, typename P>
    class expl_leapfrog: public leapfrog<H, P> {
      
    public:
      
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
      
    private:
      
    };
    
    // implicit leapfrog
    
  } // mcmc

} // stan
          

#endif
