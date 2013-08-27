#ifndef __STAN__GM__ARGUMENTS__OUTPUT__FILE__HPP__
#define __STAN__GM__ARGUMENTS__OUTPUT__FILE__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_output_file: public string_argument {
      
    public:
      
      arg_output_file(): string_argument() {
        _name = "file";
        _description = "Output file";
        _validity = "Path to existing file";
        _default = "samples.csv";
        _default_value = "samples.csv";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif