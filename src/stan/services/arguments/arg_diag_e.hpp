#ifndef STAN__SERVICES__ARGUMENTS__DIAG_E__HPP
#define STAN__SERVICES__ARGUMENTS__DIAG_E__HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_diag_e: public unvalued_argument {
      
    public:
      
      arg_diag_e() {
        
        _name = "diag_e";
        _description = "Euclidean manifold with diag metric";
        
      }
      
    };
    
  } // services
  
} // stan

#endif

