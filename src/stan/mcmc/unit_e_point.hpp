#ifndef __STAN__MCMC__UNIT_E_POINT__BETA__
#define __STAN__MCMC__UNIT_E_POINT__BETA__

#include <stan/mcmc/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with unit metric
    class unit_e_point: public ps_point {
      
    public:
      
      unit_e_point(int n, int m): ps_point(n, m), V(0), g(Eigen::VectorXd::Zero(n)) {};
      
      double V;
      Eigen::VectorXd g;
      
    };
    
  } // mcmc
  
} // stan


#endif