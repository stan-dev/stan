#ifndef STAN__GM__ARGUMENTS__FIXED__PARAMS__HPP
#define STAN__GM__ARGUMENTS__FIXED__PARAMS__HPP

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_fixed_param: public unvalued_argument {
      
    public:
      
      arg_fixed_param() {
        
        _name = "fixed_param";
        _description = "Fixed Parameter Sampler";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

