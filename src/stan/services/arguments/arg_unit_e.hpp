#ifndef STAN__SERVICES__ARGUMENTS__UNIT_E__HPP
#define STAN__SERVICES__ARGUMENTS__UNIT_E__HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_unit_e: public unvalued_argument {
      
    public:
      
      arg_unit_e() {
        
        _name = "unit_e";
        _description = "Euclidean manifold with unit metric";
        
      }
      
    };
    
  } // services
  
} // stan

#endif

