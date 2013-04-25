#ifndef __STAN__GM__ARGUMENTS__BFGS__HPP__
#define __STAN__GM__ARGUMENTS__BFGS__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_bfgs: public categorical_argument {
      
    public:
      
      arg_bfgs() {
        
        _name = "bfgs";
        _description = "BFGS with linesearch";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

