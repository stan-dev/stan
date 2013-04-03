#ifndef __STAN__MCMC__DIAG__E__POINT__BETA__
#define __STAN__MCMC__DIAG__E__POINT__BETA__

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with diagonal metric
    class diag_e_point: public ps_point {
      
    public:
      
      diag_e_point(int n, int m): ps_point(n, m),
                                  mInv(Eigen::VectorXd::Ones(n)) 
      {};
      
      Eigen::VectorXd mInv;
      
    };
    
  } // mcmc
  
} // stan


#endif