#ifndef __STAN__GM__ARGUMENTS__SOFTABS__ALPHA__HPP__
#define __STAN__GM__ARGUMENTS__SOFTABS__ALPHA__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_softabs_alpha: public real_argument {
      
    public:
      
      arg_stepsize(): real_argument() {
        _name = "alpha";
        _description = "SoftAbs regularization parameter";
        _validity = "0 < alpha";
        _default = "1";
        _default_value = 1.0;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return value > 0; }

    };
    
  } // gm
  
} // stan

#endif