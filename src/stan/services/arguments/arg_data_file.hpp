#ifndef STAN__SERVICES__ARGUMENTS__DATA__FILE__HPP
#define STAN__SERVICES__ARGUMENTS__DATA__FILE__HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_data_file: public string_argument {
      
    public:
      
      arg_data_file(): string_argument() {
        _name = "file";
        _description = "Input data file";
        _validity = "Path to existing file";
        _default = "\"\"";
        _default_value = "";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      };
      
    };
    
  } // services
  
} // stan

#endif
