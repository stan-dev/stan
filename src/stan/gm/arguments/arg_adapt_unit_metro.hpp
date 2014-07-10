#ifndef __STAN__GM__ARGUMENTS__ADAPT__UNIT__METRO__HPP__
#define __STAN__GM__ARGUMENTS__ADAPT__UNIT__METRO__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_unit_metro: public unvalued_argument {
      
    public:
      
      arg_adapt_unit_metro() {
        
        _name = "adapt_unit_metro";
        _description = "Metropolis Hastings with Adapted Unit Metric";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

