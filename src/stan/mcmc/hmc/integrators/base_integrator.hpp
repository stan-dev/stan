#ifndef __STAN__MCMC__BASE__INTEGRATOR__BETA__
#define __STAN__MCMC__BASE__INTEGRATOR__BETA__

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
