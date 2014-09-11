#ifndef STAN__SERVICES__ARGUMENTS__OPTIMIZE__ALGO__HPP
#define STAN__SERVICES__ARGUMENTS__OPTIMIZE__ALGO__HPP

#include <stan/services/arguments/list_argument.hpp>

#include <stan/services/arguments/arg_bfgs.hpp>
#include <stan/services/arguments/arg_lbfgs.hpp>
#include <stan/services/arguments/arg_newton.hpp>

namespace stan {
  
  namespace services {
    
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
    
  } // services
  
} // stan

#endif

