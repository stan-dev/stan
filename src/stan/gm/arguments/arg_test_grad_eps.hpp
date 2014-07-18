#ifndef STAN__GM__ARGUMENTS__TEST__GRAD__EPS__HPP
#define STAN__GM__ARGUMENTS__TEST__GRAD__EPS__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_test_grad_eps: public real_argument {
      
    public:
      
      arg_test_grad_eps(): real_argument() {
        _name = "epsilon";
        _description = "Finite difference step size";
        _validity = "0 < epsilon";
        _default = "1e-6";
        _default_value = 1e-6;
        _constrained = true;
        _good_value = 1e-6;
        _bad_value = -1.0;
        _value = _default_value;
      };
      
      bool is_valid(double value) { return value > 0; }

    };
    
  } // gm
  
} // stan

#endif
