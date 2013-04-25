#ifndef __STAN__GM__ARGUMENTS__METRO__HPP__
#define __STAN__GM__ARGUMENTS__METRO__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_metro: public categorical_argument {
      
    public:
      
      arg_metro() {
        
        _name = "metro";
        _description = "Random Walk Metropolis Monte Carlo";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

