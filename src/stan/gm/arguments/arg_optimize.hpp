#ifndef __STAN__GM__ARGUMENTS__OPTIMIZE__HPP__
#define __STAN__GM__ARGUMENTS__OPTIMIZE__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_nesterov.hpp>
#include <stan/gm/arguments/arg_bfgs.hpp>
#include <stan/gm/arguments/arg_newton.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_optimize: public categorical_argument {
      
    public:
      
      arg_optimize() {
        
        _name = "optimize";
        _description = "Point estimation";
        
        _subarguments.push_back(new arg_nesterov());
        _subarguments.push_back(new arg_bfgs());
        _subarguments.push_back(new arg_newton());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

