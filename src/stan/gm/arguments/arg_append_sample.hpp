#ifndef __STAN__GM__ARGUMENTS__APPEND__SAMPLE__HPP__
#define __STAN__GM__ARGUMENTS__APPEND__SAMPLE__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_append_sample: public bool_argument {
      
    public:
      
      arg_append_sample(): bool_argument() {
        _name = "append_sample";
        _description = "Append sample output to existing file?";
        _validity = "[true, false]";
        _default = "false";
        _default_value = false;
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif

