#ifndef STAN__GM__ARGUMENTS__OPTIMIZE__HPP
#define STAN__GM__ARGUMENTS__OPTIMIZE__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_optimize_algo.hpp>
#include <stan/gm/arguments/arg_iter.hpp>
#include <stan/gm/arguments/arg_save_iterations.hpp>

namespace stan {
  
  namespace gm {
    
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
    
  } // gm
  
} // stan

#endif

