#ifndef __STAN__GM__ARGUMENTS__VB__HPP__
#define __STAN__GM__ARGUMENTS__VB__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_vb: public categorical_argument {
      
    public:
      
      arg_vb() {
        
        _name = "vb";
        _description = "Variational Bayesian inference";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

