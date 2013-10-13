#ifndef __STAN__GM__ARGUMENTS__SOFTABS__HPP__
#define __STAN__GM__ARGUMENTS__SOFTABS__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_softabs_alpha.hpp>
#include <stan/gm/arguments/arg_max_num_fp.hpp>
#include <stan/gm/arguments/arg_fp_threshold.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_softabs: public categorical_argument {
      
    public:
      
      arg_softabs() {
        
        _name = "softabs";
        _description = "Riemannian manifold with SoftAbs metric";
        
        _subarguments.push_back(new arg_softabs_alpha());
        _subarguments.push_back(new arg_max_num_fp());
        _subarguments.push_back(new arg_fp_threshold());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

