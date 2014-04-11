#ifndef __STAN__GM__ARGUMENTS__LBFGS__HPP__
#define __STAN__GM__ARGUMENTS__LBFGS__HPP__

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

