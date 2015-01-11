#ifndef STAN__GM__ARGUMENTS__LBFGS__HPP
#define STAN__GM__ARGUMENTS__LBFGS__HPP

#include <stan/gm/arguments/arg_bfgs.hpp>

#include <stan/gm/arguments/arg_history_size.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_lbfgs: public arg_bfgs {
      
    public:
      
      arg_lbfgs() {
        
        _name = "lbfgs";
        _description = "LBFGS with linesearch";
        
        _subarguments.push_back(new arg_history_size());

      }
      
    };
    
  } // gm
  
} // stan

#endif

