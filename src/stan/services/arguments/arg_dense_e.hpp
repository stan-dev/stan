#ifndef STAN__SERVICES__ARGUMENTS__DENSE_E__HPP
#define STAN__SERVICES__ARGUMENTS__DENSE_E__HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace services {
    
    class arg_dense_e: public unvalued_argument {
      
    public:
      
      arg_dense_e() {
        
        _name = "dense_e";
        _description = "Euclidean manifold with dense metric";
        
      }
      
    };
    
  } // services
  
} // stan

#endif

