#ifndef __STAN__MCMC__STEPSIZE__VAR__ADAPTER__BETA__
#define __STAN__MCMC__STEPSIZE__VAR__ADAPTER__BETA__

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/var_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_var_adapter: public base_adapter {
      
    public:
      
      stepsize_var_adapter(int n, int max_adapt): _var_adaptation(n, max_adapt)
      {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return _stepsize_adaptation;
      }
      
      var_adaptation& get_var_adaptation() {
        return _var_adaptation;
      }
      
    protected:
      
      stepsize_adaptation _stepsize_adaptation;
      var_adaptation _var_adaptation;
      
    };
    
  } // mcmc
  
} // stan

#endif