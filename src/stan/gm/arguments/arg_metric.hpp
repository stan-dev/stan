#ifndef STAN__GM__ARGUMENTS__METRIC__HPP
#define STAN__GM__ARGUMENTS__METRIC__HPP

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
        
        _default_cursor = 1;
        _cursor = _default_cursor;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

