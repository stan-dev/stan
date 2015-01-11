#ifndef STAN__GM__ARGUMENTS__SAMPLE__ALGO__HPP
#define STAN__GM__ARGUMENTS__SAMPLE__ALGO__HPP

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_hmc.hpp>
#include <stan/gm/arguments/arg_fixed_param.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_sample_algo: public list_argument {
      
    public:
      
      arg_sample_algo() {
        
        _name = "algorithm";
        _description = "Sampling algorithm";
        
        _values.push_back(new arg_hmc());
        _values.push_back(new arg_fixed_param());
        
        _default_cursor = 0;
        _cursor = _default_cursor;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

