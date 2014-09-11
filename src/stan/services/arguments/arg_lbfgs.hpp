#ifndef STAN__SERVICES__ARGUMENTS__LBFGS__HPP
#define STAN__SERVICES__ARGUMENTS__LBFGS__HPP

#include <stan/services/arguments/arg_bfgs.hpp>

#include <stan/services/arguments/arg_history_size.hpp>

namespace stan {
  
  namespace services {
    
    class arg_lbfgs: public arg_bfgs {
      
    public:
      
      arg_lbfgs() {
        
        _name = "lbfgs";
        _description = "LBFGS with linesearch";
        
        _subarguments.push_back(new arg_history_size());

      }
      
    };
    
  } // services
  
} // stan

#endif

