#ifndef STAN__MCMC__BASE__ADAPTATION__BETA
#define STAN__MCMC__BASE__ADAPTATION__BETA

namespace stan {
  
  namespace mcmc {
    
    class base_adaptation {
      
    public:
      
      virtual void restart() {};
      
    };
    
  } // mcmc
  
} // stan

#endif
