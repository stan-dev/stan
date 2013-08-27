#ifndef __STAN__GM__ARGUMENTS__ADAPT__GAMMA__HPP__
#define __STAN__GM__ARGUMENTS__ADAPT__GAMMA__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_gamma: public real_argument {
      
    public:
      
      arg_adapt_gamma(): real_argument() {
        _name = "gamma";
        _description = "Adaptation regularization scale";
        _validity = "0 < gamma";
        _default = "0.05";
        _default_value = 0.05;
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

