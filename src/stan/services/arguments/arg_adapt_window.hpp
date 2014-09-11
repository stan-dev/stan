#ifndef STAN__SERVICES__ARGUMENTS__ADAPT__WINDOW__HPP
#define STAN__SERVICES__ARGUMENTS__ADAPT__WINDOW__HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_adapt_window: public u_int_argument {
      
    public:
      
      arg_adapt_window(): u_int_argument() {
        _name = "window";
        _description = "Initial width of slow adaptation interval";
        _default = "25";
        _default_value = 25;
        _value = _default_value;
      };
      
    };
    
  } // services
  
} // stan

#endif

