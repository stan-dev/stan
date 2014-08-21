#ifndef STAN__GM__ARGUMENTS__RWM__HPP
#define STAN__GM__ARGUMENTS__RWM__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_rwm: public categorical_argument {
      
    public:
      
      arg_rwm() {
        
        _name = "rwm";
        _description = "Random Walk Metropolis Monte Carlo";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

