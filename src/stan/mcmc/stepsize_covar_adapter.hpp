#ifndef __STAN__MCMC__STEPSIZE__COVAR__ADAPTER__BETA__
#define __STAN__MCMC__STEPSIZE__COVAR__ADAPTER__BETA__

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/covar_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_covar_adapter: public base_adapter {
      
    public:
      
      stepsize_covar_adapter(int n, int max_adapt): _covar_adaptation(n, max_adapt)
      {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return _stepsize_adaptation;
      }
      
      covar_adaptation& get_covar_adaptation() {
        return _covar_adaptation;
      }
      
    protected:
      
      stepsize_adaptation _stepsize_adaptation;
      covar_adaptation _covar_adaptation;
      
    };
    
  } // mcmc
  
} // stan

#endif