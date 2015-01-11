#ifndef STAN__GM__ARGUMENTS__TEST__GRADIENT__HPP
#define STAN__GM__ARGUMENTS__TEST__GRADIENT__HPP

#include <stan/gm/arguments/categorical_argument.hpp>
#include <stan/gm/arguments/arg_test_grad_eps.hpp>
#include <stan/gm/arguments/arg_test_grad_err.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_test_gradient: public categorical_argument {
      
    public:
      
      arg_test_gradient() {
        
        _name = "gradient";
        _description = "Check model gradient against finite differences";
        
        _subarguments.push_back(new arg_test_grad_eps());
        _subarguments.push_back(new arg_test_grad_err());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

