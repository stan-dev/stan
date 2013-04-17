#ifndef __STAN__MCMC__STATIC__ADAPTER__BASE__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__BASE__BETA__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace mcmc {
    
    class base_adapter {
    
    public:
      
      virtual void init() = 0;

    };
    
  } // mcmc
  
} // stan

#endif