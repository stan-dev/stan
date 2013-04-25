#ifndef __STAN__GM__ARGUMENTS__METRIC__HPP__
#define __STAN__GM__ARGUMENTS__METRIC__HPP__

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_unit_e.hpp>
#include <stan/gm/arguments/arg_diag_e.hpp>
#include <stan/gm/arguments/arg_dense_e.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_metric: public list_argument {
      
    public:
      
      arg_metric() {
        
        _name = "metric";
        _description = "Geometry of base manifold";
        
        _values.push_back(new arg_unit_e());
        _values.push_back(new arg_diag_e());
        _values.push_back(new arg_dense_e());
        
        _default_cursor = 0;
        _cursor = 0;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

