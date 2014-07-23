#ifndef STAN__GM__ARGUMENTS__RANDOM__HPP
#define STAN__GM__ARGUMENTS__RANDOM__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_seed.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_random: public categorical_argument {
      
    public:
      
      arg_random() {
        
        _name = "random";
        _description = "Random number configuration";
        
        _subarguments.push_back(new arg_seed());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

