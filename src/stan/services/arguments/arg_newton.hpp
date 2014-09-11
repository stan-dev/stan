#ifndef STAN__SERVICES__ARGUMENTS__NEWTON__HPP
#define STAN__SERVICES__ARGUMENTS__NEWTON__HPP

#include <stan/services/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_newton: public categorical_argument {
      
    public:
      
      arg_newton() {
        
        _name = "newton";
        _description = "Newton's method";
        
      }
      
    };
    
  } // services
  
} // stan

#endif

