#ifndef STAN__GM__ARGUMENTS__DIAG_E__HPP
#define STAN__GM__ARGUMENTS__DIAG_E__HPP

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diag_e: public unvalued_argument {
      
    public:
      
      arg_diag_e() {
        
        _name = "diag_e";
        _description = "Euclidean manifold with diag metric";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

