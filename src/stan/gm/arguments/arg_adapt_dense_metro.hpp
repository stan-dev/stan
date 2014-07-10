#ifndef __STAN__GM__ARGUMENTS__ADAPT__DENSE__METRO__HPP__
#define __STAN__GM__ARGUMENTS__ADAPT__DENSE__METRO__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_dense_metro: public unvalued_argument {
      
    public:
      
      arg_adapt_dense_metro() {
        
        _name = "adapt_dense_metro";
        _description = "Metropolis Hastings with Adapted Dense Metric";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

