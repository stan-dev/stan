#ifndef STAN__GM__ARGUMENTS__RESUME__SAVE__TO__FILE__HPP
#define STAN__GM__ARGUMENTS__RESUME__SAVE__TO__FILE__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_resume_save_to_file: public string_argument {
      
    public:
      
      arg_resume_save_to_file(): string_argument() {
        _name = "save_to_file";
        _description = "Name of the file where sampling resuming information will be saved to";
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
