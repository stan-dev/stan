#ifndef __STAN__GM__ARGUMENTS__DIAGNOSTIC__FILE__HPP__
#define __STAN__GM__ARGUMENTS__DIAGNOSTIC__FILE__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diagnostic_file: public string_argument {
      
    public:
      
      arg_diagnostic_file(): string_argument() {
        _name = "diagnostic";
        _description = "Output file for diagnostic information";
        _default = "\"\"";
        _default_value = "";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif