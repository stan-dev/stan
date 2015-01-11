#ifndef STAN__GM__ARGUMENTS__DIAGNOSTIC__FILE__HPP
#define STAN__GM__ARGUMENTS__DIAGNOSTIC__FILE__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diagnostic_file: public string_argument {
      
    public:
      
      arg_diagnostic_file(): string_argument() {
        _name = "diagnostic_file";
        _description = "Auxiliary output file for diagnostic information";
        _validity = "Path to existing file";
        _default = "\"\"";
        _default_value = "";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
