#ifndef __STAN__GM__ARGUMENTS__NEWTON__HPP__
#define __STAN__GM__ARGUMENTS__NEWTON__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_newton: public categorical_argument {
      
    public:
      
      arg_newton() {
        
        _name = "newton";
        _description = "Newton's method";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

