#ifndef STAN__MCMC__DIAG__E__POINT__BETA
#define STAN__MCMC__DIAG__E__POINT__BETA

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with diagonal metric
    class diag_e_point: public ps_point {
      
    public:
      
      diag_e_point(int n): ps_point(n), mInv(n) {
        mInv.setOnes();
      };
      
      Eigen::VectorXd mInv;
      
      diag_e_point(const diag_e_point& z): ps_point(z), mInv(z.mInv.size()) {
        fast_vector_copy_<double>(mInv, z.mInv);
      }
      
      void write_metric(std::ostream* o) {
        if(!o) return;
        *o << "# Diagonal elements of inverse mass matrix:" << std::endl;
        *o << "# " << mInv(0) << std::flush;
        for (int i = 1; i < mInv.size(); ++i)
          *o << ", " << mInv(i) << std::flush;
        *o << std::endl;
      };
      
    };
    
  } // mcmc
  
} // stan


#endif
