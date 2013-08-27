#ifndef __STAN__GM__ARGUMENTS__TEST__GRADIENT__HPP__
#define __STAN__GM__ARGUMENTS__TEST__GRADIENT__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_test_gradient: public unvalued_argument {
      
    public:
      
      arg_test_gradient() {
        
        _name = "gradient";
        _description = "Check model gradient against finite differences";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

