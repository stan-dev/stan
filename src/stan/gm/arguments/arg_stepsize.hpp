#ifndef __STAN__GM__ARGUMENTS__STEPSIZE__HPP__
#define __STAN__GM__ARGUMENTS__STEPSIZE__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_stepsize: public real_argument {
      
    public:
      
      arg_stepsize(): real_argument() {
        _name = "stepsize";
        _description = "Step size for discrete evolution";
        _validity = "0 < stepsize";
        _default = "1";
      };
      
      bool is_valid(double value) { return value > 0; }

    };
    
  } // gm
  
} // stan

#endif