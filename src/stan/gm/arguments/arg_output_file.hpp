#ifndef STAN__GM__ARGUMENTS__OUTPUT__FILE__HPP
#define STAN__GM__ARGUMENTS__OUTPUT__FILE__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_output_file: public string_argument {
      
    public:
      
      arg_output_file(): string_argument() {
        _name = "file";
        _description = "Output file";
        _validity = "Path to existing file";
        _default = "output.csv";
        _default_value = "output.csv";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
