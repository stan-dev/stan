#ifndef __STAN__GM__ARGUMENTS__INIT__HPP__
#define __STAN__GM__ARGUMENTS__INIT__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_init: public string_argument {
      
    public:
      
      arg_init(): string_argument() {
        _name = "init";
        _description = std::string("Initialization method:")
          + std::string("\"0\" initializes to zero,")
          + std::string("\"random\" initializes randomly,")
          + std::string("anything else identifies a file");
        _default = "\"random\"";
        _default_value = "random";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
