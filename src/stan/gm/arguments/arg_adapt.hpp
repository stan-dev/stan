#ifndef __STAN__GM__ARGUMENTS__ADAPT__HPP__
#define __STAN__GM__ARGUMENTS__ADAPT__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_adapt_gamma.hpp>
#include <stan/gm/arguments/arg_adapt_delta.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt: public categorical_argument {
      
    public:
      
      arg_adapt() {
        
        _name = "adapt";
        _description = "Warmup Adaptation";
        
        _subarguments.push_back(new arg_adapt_gamma());
        _subarguments.push_back(new arg_adapt_delta());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

