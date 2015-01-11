#ifndef STAN__GM__ARGUMENTS__NUTS__HMC__HPP
#define STAN__GM__ARGUMENTS__NUTS__HMC__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_max_depth.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_nuts: public categorical_argument {
      
    public:
      
      arg_nuts() {
        
        _name = "nuts";
        _description = "The No-U-Turn Sampler";
        
        _subarguments.push_back(new arg_max_depth());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

