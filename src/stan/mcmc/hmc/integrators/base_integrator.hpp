#ifndef __STAN__MCMC__BASE__INTEGRATOR__BETA__
#define __STAN__MCMC__BASE__INTEGRATOR__BETA__

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class base_integrator {
      
    public:
      
      virtual void evolve(P& z, H& hamiltonian, const double epsilon) = 0;
      
    };
    
  } // mcmc

} // stan
          

#endif
