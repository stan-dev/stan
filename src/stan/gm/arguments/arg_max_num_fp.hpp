#ifndef __STAN__GM__ARGUMENTS__MAX__NUM__FP__HPP__
#define __STAN__GM__ARGUMENTS__MAX__NUM__FP__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_max_num_fp: public int_argument {
      
    public:
      
      arg_max_num_fp(): int_argument() {
        _name = "max_fp";
        _description = "Maximum number of floating interations";
        _validity = "0 < max_fp";
        _default = "50";
        _default_value = 50.0;
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