#ifndef STAN__MCMC__EXPL__LEAPFROG__BETA
#define STAN__MCMC__EXPL__LEAPFROG__BETA

#include <stan/math/matrix/Eigen.hpp>
#include <stan/mcmc/hmc/integrators/base_leapfrog.hpp>

namespace stan {
  
  namespace mcmc {
    
    template <typename H, typename P>
    class expl_leapfrog: public base_leapfrog<H, P> {
      
    public:
      
      expl_leapfrog(std::ostream* o=0): base_leapfrog<H, P>(o) {};
      
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
    
  } // mcmc
  
} // stan


#endif
