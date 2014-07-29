#ifndef STAN__GM__ARGUMENTS__DENSE_E__HPP
#define STAN__GM__ARGUMENTS__DENSE_E__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_dense_e: public categorical_argument {
      
    public:
      
      arg_dense_e() {
        
        _name = "dense_e";
        _description = "Euclidean manifold with dense metric";
        
        _subarguments.push_back(new arg_file());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

