#ifndef __STAN__GM__ARGUMENTS__DENSE_E__HPP__
#define __STAN__GM__ARGUMENTS__DENSE_E__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_dense_e: public unvalued_argument {
      
    public:
      
      arg_dense_e() {
        
        _name = "dense_e";
        _description = "Euclidean manifold with dense metric";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

