#ifndef STAN__SERVICES__ARGUMENTS__ADAPT__TERM__BUFFER__HPP
#define STAN__SERVICES__ARGUMENTS__ADAPT__TERM__BUFFER__HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_adapt_term_buffer: public u_int_argument {
      
    public:
      
      arg_adapt_term_buffer(): u_int_argument() {
        _name = "term_buffer";
        _description = std::string("Width of final fast adaptation interval");
        _default = "50";
        _default_value = 50;
        _value = _default_value;
      };
      
    };
    
  } // services
  
} // stan

#endif

