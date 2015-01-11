#ifndef STAN__GM__ARGUMENTS__OPTIMIZE__ALGO__HPP
#define STAN__GM__ARGUMENTS__OPTIMIZE__ALGO__HPP

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_bfgs.hpp>
#include <stan/gm/arguments/arg_lbfgs.hpp>
#include <stan/gm/arguments/arg_newton.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_optimize_algo: public list_argument {
      
    public:
      
      arg_optimize_algo() {
        
        _name = "algorithm";
        _description = "Optimization algorithm";
        
        _values.push_back(new arg_bfgs());
        _values.push_back(new arg_lbfgs());
        _values.push_back(new arg_newton());
        
        _default_cursor = 1;
        _cursor = _default_cursor;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

