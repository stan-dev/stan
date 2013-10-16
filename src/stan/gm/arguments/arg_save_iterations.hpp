#ifndef __STAN__GM__ARGUMENTS__OUTPUT__SAVE__ITERATIONS__HPP__
#define __STAN__GM__ARGUMENTS__OUTPUT__SAVE__ITERATIONS__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_save_iterations: public bool_argument {
      
    public:
      
      arg_save_iterations(): bool_argument() {
        _name = "save_iterations";
        _description = "Stream optimization progress to output?";
        _validity = "[0, 1]";
        _default = "0";
        _default_value = false;
        _constrained = false;
        _good_value = 1;
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif

