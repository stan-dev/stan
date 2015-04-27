#ifndef STAN__SERVICES__ARGUMENTS__OPTIMIZE__HPP
#define STAN__SERVICES__ARGUMENTS__OPTIMIZE__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_optimize_algo.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_save_iterations.hpp>

namespace stan {
  
  namespace services {
    
    class arg_optimize: public categorical_argument {
      
    public:
      
      arg_optimize() {
        
        _name = "optimize";
        _description = "Point estimation";
        
        _subarguments.push_back(new arg_optimize_algo());
        _subarguments.push_back(new arg_iter());
        _subarguments.push_back(new arg_save_iterations());
        
      }
      
    };
    
  } // services
  
} // stan

#endif

