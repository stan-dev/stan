#ifndef STAN__GM__ARGUMENTS__STATIC__HMC__HPP
#define STAN__GM__ARGUMENTS__STATIC__HMC__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_int_time.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_static: public categorical_argument {
      
    public:
      
      arg_static() {
        
        _name = "static";
        _description = "Static integration time";
        
        _subarguments.push_back(new arg_int_time());
        
      }
 
    };
    
  } // gm
  
} // stan

#endif

