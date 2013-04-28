#ifndef __STAN__GM__ARGUMENTS__NESTEROV__HPP__
#define __STAN__GM__ARGUMENTS__NESTEROV__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_stepsize.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_nesterov: public categorical_argument {
      
    public:
      
      arg_nesterov() {
        
        _name = "nesterov";
        _description = "Nesterov's accelerated gradient method";
        
        _subarguments.push_back(new arg_stepsize());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

