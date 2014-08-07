#ifndef STAN__GM__ARGUMENTS__REFRESH__HPP
#define STAN__GM__ARGUMENTS__REFRESH__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_refresh: public int_argument {
      
    public:
      
      arg_refresh(): int_argument() {
        _name = "refresh";
        _description = "Number of interations between screen updates";
        _validity = "0 <= refresh";
        _default = "100";
        _default_value = 100;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(int value) { return value >= 0; }
      
    };
    
  } // gm
  
} // stan

#endif

