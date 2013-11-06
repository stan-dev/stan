#ifndef __STAN__GM__ARGUMENTS__OPTIMIZE__ALGO__HPP__
#define __STAN__GM__ARGUMENTS__OPTIMIZE__ALGO__HPP__

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_nesterov.hpp>
#include <stan/gm/arguments/arg_bfgs.hpp>
#include <stan/gm/arguments/arg_newton.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_optimize_algo: public list_argument {
      
    public:
      
      arg_optimize_algo() {
        
        _name = "algorithm";
        _description = "Optimization algorithm";
        
        _values.push_back(new arg_nesterov());
        _values.push_back(new arg_bfgs());
        _values.push_back(new arg_newton());
        
        _default_cursor = 1;
        _cursor = _default_cursor;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

