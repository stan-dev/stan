#ifndef STAN__GM__ARGUMENTS__DIAG_E__HPP
#define STAN__GM__ARGUMENTS__DIAG_E__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_diag_e: public categorical_argument {
      
    public:
      
      arg_diag_e() {
        
        _name = "diag_e";
        _description = "Euclidean manifold with diag metric";
        
        _subarguments.push_back(new arg_file());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

