#ifndef STAN__GM__ARGUMENTS__INIT__HPP
#define STAN__GM__ARGUMENTS__INIT__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_init: public string_argument {
      
    public:
      
      arg_init(): string_argument() {
        _name = "init";
        _description = std::string("Initialization method: ")
          + std::string("\"x\" initializes randomly bewteen [-x, x], ")
          + std::string("\"0\" initializes to 0, ")
          + std::string("anything else identifies a file of values");
        _default = "\"2\"";
        _default_value = "2";
        _constrained = false;
        _good_value = "../src/test/test-models/test_model.init.R";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
