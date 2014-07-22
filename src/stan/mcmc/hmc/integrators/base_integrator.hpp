#ifndef STAN__MCMC__BASE__INTEGRATOR__BETA
#define STAN__MCMC__BASE__INTEGRATOR__BETA

#include <ostream>

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class base_integrator {
      
    public:
      
      base_integrator(std::ostream* o): out_stream_(o) {};
      
      virtual void evolve(P& z, H& hamiltonian, const double epsilon) = 0;
      
    protected:
      
      std::ostream* out_stream_;
      
    };
    
  } // mcmc

} // stan
          

#endif
