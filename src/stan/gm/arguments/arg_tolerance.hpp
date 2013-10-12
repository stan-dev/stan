#ifndef __STAN__GM__ARGUMENTS__TOLERANCE_HPP__
#define __STAN__GM__ARGUMENTS__TOLERANCE_HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_tolerance : public real_argument {
      
    public:
      
      arg_tolerance(const char *name, const char *desc, const char *def_str, double def) : real_argument() {
        _name = name;
        _description = desc;
        _validity = "0 < tol";
        _default = def_str;
        _default_value = def;
        _constrained = true;
        _good_value = 1.0;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return value > 0; }

    };
    
  } // gm
  
} // stan

#endif
