#ifndef STAN__GM__ARGUMENTS__NUM__WARMUP__HPP
#define STAN__GM__ARGUMENTS__NUM__WARMUP__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_num_warmup: public int_argument {
      
    public:
      
      arg_num_warmup(): int_argument() {
        _name = "num_warmup";
        _description = "Number of warmup iterations";
        _validity = "0 <= warmup";
        _default = "1000";
        _default_value = 1000;
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

