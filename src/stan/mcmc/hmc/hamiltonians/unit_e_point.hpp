#ifndef STAN__MCMC__UNIT__E__POINT__BETA
#define STAN__MCMC__UNIT__E__POINT__BETA

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with unit metric
    class unit_e_point: public ps_point {
      
    public:
      
      unit_e_point(int n): ps_point(n) {};

    };
    
  } // mcmc
  
} // stan


#endif
