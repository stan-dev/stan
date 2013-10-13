#ifndef __STAN__GM__ARGUMENTS__FP__THRESHOLD__HPP__
#define __STAN__GM__ARGUMENTS__FP__THRESHOLD__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_fp_threshold: public real_argument {
      
    public:
      
      arg_fp_threshold(): real_argument() {
        _name = "fp_threshold";
        _description = "Threshold for terminating floating point iterations";
        _validity = "0 < fp_threshold";
        _default = "1e-8";
        _default_value = 1e-8;
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