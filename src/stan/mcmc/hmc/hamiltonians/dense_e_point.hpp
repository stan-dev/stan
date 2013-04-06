#ifndef __STAN__MCMC__DENSE__E__POINT__BETA__
#define __STAN__MCMC__DENSE__E__POINT__BETA__

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with dense metric
    class dense_e_point: public ps_point {
      
    public:
      
      dense_e_point(int n, int m): ps_point(n, m),
                                   mInv(Eigen::MatrixXd::Identity(n, n)) 
      {};
      
      Eigen::MatrixXd mInv;
      
      void write_metric(std::ostream& o) {
        //o << "# Inverse mass matrix elements:" << std::endl;
        o << "# Elements of inverse mass matrix:" << std::endl;
        for(size_t i = 0; i < mInv.rows(); ++i) {
          o << "# " << mInv(i, 0) << std::flush;
          for(size_t j = 1; j < mInv.cols(); ++j)
            o << ", " << mInv(i, j) << std::flush;
          o << std::endl;
        }
      };
      
    };
    
  } // mcmc
  
} // stan


#endif