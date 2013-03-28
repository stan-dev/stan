#ifndef __STAN__MCMC__BASE__LEAPFROG__BETA__
#define __STAN__MCMC__BASE__LEAPFROG__BETA__

#include <stan/mcmc/base_integrator.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename H, typename P>
    class base_leapfrog: public base_integrator<H, P> {
      
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
      
    };
    
  } // mcmc
  
} // stan


#endif
