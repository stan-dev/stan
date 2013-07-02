#ifndef __STAN__GM__ARGUMENTS__DATA__HPP__
#define __STAN__GM__ARGUMENTS__DATA__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_data: public string_argument {
      
    public:
      
      arg_data(): string_argument() {
        _name = "data";
        _short_description = "Input data file";
        _description = "Input data file";
        _default = "\"\"";
        _default_value = "";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
