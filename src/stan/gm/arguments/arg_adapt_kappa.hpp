#ifndef STAN__GM__ARGUMENTS__ADAPT__KAPPA__HPP
#define STAN__GM__ARGUMENTS__ADAPT__KAPPA__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_kappa: public real_argument {
      
    public:
      
      arg_adapt_kappa(): real_argument() {
        _name = "kappa";
        _description = "Adaptation relaxation exponent";
        _validity = "0 < kappa";
        _default = "0.75";
        _default_value = 0.75;
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

