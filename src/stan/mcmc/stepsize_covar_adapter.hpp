#ifndef __STAN__MCMC__STEPSIZE__COVAR__ADAPTER__BETA__
#define __STAN__MCMC__STEPSIZE__COVAR__ADAPTER__BETA__

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/covar_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_covar_adapter: public base_adapter {
      
    public:
      
      stepsize_covar_adapter(int n): _covar_adaptation(n)
      {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return _stepsize_adaptation;
      }
      
      covar_adaptation& get_covar_adaptation() {
        return _covar_adaptation;
      }
      
      void set_window_params(unsigned int num_warmup,
                             unsigned int init_buffer,
                             unsigned int term_buffer,
                             unsigned int base_window,
                             std::ostream* e = 0) {
        _covar_adaptation.set_window_params(num_warmup,
                                             init_buffer,
                                             term_buffer,
                                             base_window,
                                             e);
      }
      
    protected:
      
      stepsize_adaptation _stepsize_adaptation;
      covar_adaptation _covar_adaptation;
      
    };
    
  } // mcmc
  
} // stan

#endif