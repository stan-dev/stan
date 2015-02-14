#ifndef STAN__SERVICES__ARGUMENTS__DIAGNOSE__HPP
#define STAN__SERVICES__ARGUMENTS__DIAGNOSE__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_test.hpp>

namespace stan {
  
  namespace services {
    
    class arg_diagnose: public categorical_argument {
      
    public:
      
      arg_diagnose() {
        
        _name = "diagnose";
        _description = "Model diagnostics";
        
        _subarguments.push_back(new arg_test());
        
      }
      
    };
    
  } // services
  
} // stan

#endif

