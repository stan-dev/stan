#ifndef STAN__MCMC__DENSE__E__POINT__BETA
#define STAN__MCMC__DENSE__E__POINT__BETA

#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>

namespace stan {
  
  namespace mcmc {
    
    // Point in a phase space with a base
    // Euclidean manifold with dense metric
    class dense_e_point: public ps_point {
      
    public:
      
      dense_e_point(int n): ps_point(n), mInv(n, n) {
        mInv.setIdentity();
      };
      
      Eigen::MatrixXd mInv;
      
      dense_e_point(const dense_e_point& z): ps_point(z), mInv(z.mInv.rows(), z.mInv.cols()) {
        fast_matrix_copy_<double>(mInv, z.mInv);
      }
      
      void write_metric(std::ostream* o) {
        if(!o) return;
        *o << "# Elements of inverse mass matrix:" << std::endl;
        for(int i = 0; i < mInv.rows(); ++i) {
          *o << "# " << mInv(i, 0) << std::flush;
          for(int j = 1; j < mInv.cols(); ++j)
            *o << ", " << mInv(i, j) << std::flush;
          *o << std::endl;
        }
      };
      
    };
    
  } // mcmc
  
} // stan


#endif
