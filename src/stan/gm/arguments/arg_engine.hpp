#ifndef STAN__GM__ARGUMENTS__ENGINE__HPP
#define STAN__GM__ARGUMENTS__ENGINE__HPP

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_static.hpp>
#include <stan/gm/arguments/arg_nuts.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_engine: public list_argument {
      
    public:
      
      arg_engine() {
        
        _name = "engine";
        _description = "Engine for Hamiltonian Monte Carlo";
        
        _values.push_back(new arg_static());
        _values.push_back(new arg_nuts());
        
        _default_cursor = 1;
        _cursor = _default_cursor;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

