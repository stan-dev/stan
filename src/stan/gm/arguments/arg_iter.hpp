#ifndef STAN__GM__ARGUMENTS__ITER__HPP
#define STAN__GM__ARGUMENTS__ITER__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_iter: public int_argument {
      
    public:
      
      arg_iter(): int_argument() {
        _name = "iter";
        _description = "Total number of iterations";
        _validity = "0 < iter";
        _default = "2000";
        _default_value = 2000;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(int value) { return value > 0; }
      
    };
    
  } // gm
  
} // stan

#endif

