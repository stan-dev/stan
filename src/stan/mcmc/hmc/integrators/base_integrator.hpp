#ifndef STAN__MCMC__BASE__INTEGRATOR__BETA
#define STAN__MCMC__BASE__INTEGRATOR__BETA

#include <ostream>

namespace stan {

  namespace mcmc {

    template <typename H, typename P>
    class base_integrator {
      
    public:
      
      base_integrator(std::ostream* o): _out_stream(o) {};
      
      virtual void evolve(P& z, H& hamiltonian, const double epsilon) = 0;
      
    protected:
      
      std::ostream* _out_stream;
      
    };
    
  } // mcmc

} // stan
          

#endif
