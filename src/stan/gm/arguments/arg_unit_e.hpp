#ifndef __STAN__GM__ARGUMENTS__UNIT_E__HPP__
#define __STAN__GM__ARGUMENTS__UNIT_E__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_unit_e: public unvalued_argument {
      
    public:
      
      arg_unit_e() {
        
        _name = "unit_e";
        _description = "Euclidean manifold with unit metric";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

