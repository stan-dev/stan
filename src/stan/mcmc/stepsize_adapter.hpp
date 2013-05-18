#ifndef __STAN__MCMC__STEPSIZE__ADAPTER__BETA__
#define __STAN__MCMC__STEPSIZE__ADAPTER__BETA__

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_adapter: public base_adapter {
      
    public:
      
      stepsize_adapter() {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return _stepsize_adaptation;
      }
      
    protected:
      
      stepsize_adaptation _stepsize_adaptation;
      
    };
    
  } // mcmc
  
} // stan

#endif