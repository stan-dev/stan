#ifndef __STAN__GM__ARGUMENTS__OUTPUT__HESSIAN__HPP__
#define __STAN__GM__ARGUMENTS__OUTPUT__HESSIAN__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_output_hessian: public string_argument {
      
    public:
      
      arg_output_hessian(): string_argument() {
        _name = "hessian";
        _description = "Output CSV file to dump the Hessian";
        _validity = "Path to file in existing directory";
        _default = "";
        _default_value = "";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif
