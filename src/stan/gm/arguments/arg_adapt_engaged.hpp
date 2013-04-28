#ifndef __STAN__GM__ARGUMENTS__ADAPT__ENGAGED__HPP__
#define __STAN__GM__ARGUMENTS__ADAPT__ENGAGED__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_engaged: public bool_argument {
      
    public:
      
      arg_adapt_engaged(): bool_argument() {
        _name = "engaged";
        _description = "Adaptation engaged?";
        _validity = "[0, 1]";
        _default = "0";
        _default_value = false;
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif

