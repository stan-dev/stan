#ifndef STAN__GM__ARGUMENTS__NUM__SAMPLES__HPP
#define STAN__GM__ARGUMENTS__NUM__SAMPLES__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_num_samples: public int_argument {
      
    public:
      
      arg_num_samples(): int_argument() {
        _name = "num_samples";
        _description = "Number of sampling iterations";
        _validity = "0 <= num_samples";
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

