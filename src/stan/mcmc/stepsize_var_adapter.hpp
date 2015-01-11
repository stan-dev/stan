#ifndef STAN__MCMC__STEPSIZE__VAR__ADAPTER__BETA
#define STAN__MCMC__STEPSIZE__VAR__ADAPTER__BETA

#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/var_adaptation.hpp>

namespace stan {
  
  namespace mcmc {
    
    class stepsize_var_adapter: public base_adapter {
      
    public:
      
      stepsize_var_adapter(int n): var_adaptation_(n)
      {};
      
      stepsize_adaptation& get_stepsize_adaptation() {
        return stepsize_adaptation_;
      }
      
      var_adaptation& get_var_adaptation() {
        return var_adaptation_;
      }
      
      void set_window_params(unsigned int num_warmup,
                             unsigned int init_buffer,
                             unsigned int term_buffer,
                             unsigned int base_window,
                             std::ostream* e = 0) {
        var_adaptation_.set_window_params(num_warmup,
                                          init_buffer,
                                          term_buffer,
                                          base_window,
                                          e);
      }
      
      
    protected:
      
      stepsize_adaptation stepsize_adaptation_;
      var_adaptation var_adaptation_;
      
    };
    
  } // mcmc
  
} // stan

#endif
