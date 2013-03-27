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
      
      virtual void evolve(ps_point& z, H& hamiltonian, const double epsilon) = 0;
      
    };
    
    template <typename H>
    class leapfrog: public integrator<H> {
      
    public:
      
      void evolve(ps_point& z, H& hamiltonian, const double epsilon) {

        begin_update_p(z, hamiltonian, 0.5 * epsilon);
        
        update_q(z, hamiltonian, epsilon);
        hamiltonian.update(z);
        
        end_update_p(z, hamiltonian, 0.5 * epsilon);
        
      }
      
      virtual void begin_update_p(ps_point& z, H hamiltonian, double epsilon) = 0;
      virtual void update_q(ps_point& z, H hamiltonian, double epsilon) = 0;
      virtual void end_update_p(ps_point& z, H hamiltonian, double epsilon) = 0;
      
    protected:
      
    };
    
    template <typename H>
    class expl_leapfrog: public leapfrog<H> {
      
    public:
      
      void begin_update_p(ps_point& z, H hamiltonian, double epsilon) { 
        z.p -= epsilon * hamiltonian.dphi_dq(z); 
      }
      
      void update_q(ps_point& z, H hamiltonian, double epsilon) { 
        Eigen::Map<Eigen::VectorXd> q(&(z.q[0]), z.q.size());
        q += epsilon * hamiltonian.dtau_dp(z); 
      }
      
      void end_update_p(ps_point& z, H hamiltonian, double epsilon) { 
        z.p -= epsilon * hamiltonian.dphi_dq(z); 
      }
      
    private:
      
    };
    
    // implicit leapfrog
    
  } // mcmc

} // stan
          

#endif
