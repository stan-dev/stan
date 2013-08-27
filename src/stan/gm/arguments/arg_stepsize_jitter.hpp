#ifndef __STAN__GM__ARGUMENTS__STEPSIZE__JITTER__HPP__
#define __STAN__GM__ARGUMENTS__STEPSIZE__JITTER__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_stepsize_jitter: public real_argument {
      
    public:
      
      arg_stepsize_jitter(): real_argument() {
        _name = "stepsize_jitter";
        _description = "Uniformly random jitter of the stepsize, in percent";
        _validity = "0 <= stepsize_jitter <= 1";
        _default = "0";
        _default_value = 0.0;
        _constrained = true;
        _good_value = 0.5;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return 0 <= value && value <= 1; }
      
    };
    
  } // gm
  
} // stan

#endif