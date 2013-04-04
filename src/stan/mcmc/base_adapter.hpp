#ifndef __STAN__MCMC__STATIC__ADAPTER__BASE__BETA__
#define __STAN__MCMC__STATIC__ADAPTER__BASE__BETA__

#include <Eigen/Dense>

namespace stan {
  
  namespace mcmc {
    
    class base_adapter {
    
    public:
      
      virtual void init() = 0;

    };
    
  } // mcmc
  
} // stan

#endif