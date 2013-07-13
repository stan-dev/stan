#ifndef __STAN__GM__ARGUMENTS__DIAGNOSTIC__HPP__
#define __STAN__GM__ARGUMENTS__DIAGNOSTIC__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_test.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diagnostic: public categorical_argument {
      
    public:
      
      arg_diagnostic() {
        
        _name = "diagnostic";
        _description = "Model diagnostics";
        
        _subarguments.push_back(new arg_test());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

