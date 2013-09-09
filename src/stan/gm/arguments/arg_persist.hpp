#ifndef __STAN__GM__ARGUMENTS__PERSIST__HPP__
#define __STAN__GM__ARGUMENTS__PERSIST__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_persist: public unvalued_argument {
      
    public:
      
      arg_persist() {
        
        _name = "persist";
        _description = "Persistent Sampler (leaves state constant)";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

