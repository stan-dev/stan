#ifndef STAN__GM__ARGUMENTS__RESUME__LOAD__FROM__FILE__HPP
#define STAN__GM__ARGUMENTS__RESUME__LOAD__FROM__FILE__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_resume_load_from_file: public string_argument {
      
    public:
      
      arg_resume_load_from_file(): string_argument() {
        _name = "load_from_file";
        _description = "Name of the file containing sampling resuming information";
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
