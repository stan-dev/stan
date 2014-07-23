#ifndef STAN__GM__ARGUMENTS__ADAPT__DELTA__HPP
#define STAN__GM__ARGUMENTS__ADAPT__DELTA__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_adapt_delta: public real_argument {
      
    public:
      
      arg_adapt_delta(): real_argument() {
        _name = "delta";
        _description = "Adaptation target acceptance statistic";
        _validity = "0 < delta < 1";
        _default = "0.8";
        _default_value = 0.8;
        _constrained = true;
        _good_value = 0.5;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return 0 < value && value < 1; }
      
    };
    
  } // gm
  
} // stan

#endif
