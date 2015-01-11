#ifndef STAN__MCMC__STEPSIZE__ADAPTER__BETA
#define STAN__MCMC__STEPSIZE__ADAPTER__BETA

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_adapter: public base_adapter {
      
    public:
      
      stepsize_adapter() {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return stepsize_adaptation_;
      }
      
    protected:
      
      stepsize_adaptation stepsize_adaptation_;
      
    };
    
  } // mcmc
  
} // stan

#endif
