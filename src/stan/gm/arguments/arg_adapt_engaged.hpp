#ifndef STAN__GM__ARGUMENTS__ADAPT__ENGAGED__HPP
#define STAN__GM__ARGUMENTS__ADAPT__ENGAGED__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_engaged: public bool_argument {
      
    public:
      
      arg_adapt_engaged(): bool_argument() {
        _name = "engaged";
        _description = "Adaptation engaged?";
        _validity = "[0, 1]";
        _default = "1";
        _default_value = true;
        _constrained = false;
        _good_value = 1;
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif

