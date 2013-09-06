#ifndef __STAN__GM__ARGUMENTS__FAIL__HPP__
#define __STAN__GM__ARGUMENTS__FAIL__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_fail: public unvalued_argument {
      
    public:
      
      arg_fail() {
        
        _name = "fail";
        _description = "Dummy argument to induce failures for testing";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

