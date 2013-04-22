#ifndef __STAN__MCMC__BASE__ADAPTATION__BETA__
#define __STAN__MCMC__BASE__ADAPTATION__BETA__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  
  namespace mcmc {
    
    class base_adaptation {
      
    public:
      
      virtual void restart() {};
      
    };
    
  } // mcmc
  
} // stan

#endif