#ifndef STAN__GM__ARGUMENTS__THIN__HPP
#define STAN__GM__ARGUMENTS__THIN__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_thin: public int_argument {
      
    public:
      
      arg_thin(): int_argument() {
        _name = "thin";
        _description = "Period between saved samples";
        _validity = "0 < thin";
        _default = "1";
        _default_value = 1;
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

