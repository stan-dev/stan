#ifndef STAN__GM__ARGUMENTS__DIAGNOSE__HPP
#define STAN__GM__ARGUMENTS__DIAGNOSE__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_test.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diagnose: public categorical_argument {
      
    public:
      
      arg_diagnose() {
        
        _name = "diagnose";
        _description = "Model diagnostics";
        
        _subarguments.push_back(new arg_test());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

