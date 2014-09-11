#ifndef STAN__SERVICES__ARGUMENTS__FIXED__PARAMS__HPP
#define STAN__SERVICES__ARGUMENTS__FIXED__PARAMS__HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_fixed_param: public unvalued_argument {
      
    public:
      
      arg_fixed_param() {
        
        _name = "fixed_param";
        _description = "Fixed Parameter Sampler";
        
      }
      
    };
    
  } // services
  
} // stan

#endif

