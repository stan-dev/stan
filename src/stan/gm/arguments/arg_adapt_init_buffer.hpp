#ifndef STAN__GM__ARGUMENTS__ADAPT__INIT__BUFFER__HPP
#define STAN__GM__ARGUMENTS__ADAPT__INIT__BUFFER__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_init_buffer: public u_int_argument {
      
    public:
      
      arg_adapt_init_buffer(): u_int_argument() {
        _name = "init_buffer";
        _description = std::string("Width of initial fast adaptation interval");
        _default = "75";
        _default_value = 75;
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif

