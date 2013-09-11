#ifndef __STAN__GM__ARGUMENTS__TEST__TRAJECTORY__HPP__
#define __STAN__GM__ARGUMENTS__TEST__TRAJECTORY__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_stepsize.hpp>
#include <stan/gm/arguments/arg_int_time.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_test_trajectory: public categorical_argument {
      
    public:
      
      arg_test_trajectory() {
        
        _name = "trajectory";
        _description = "Trajectory diagnostic";
        
        _subarguments.push_back(new arg_stepsize());
        _subarguments.push_back(new arg_int_time());
        
      }
 
    };
    
  } // gm
  
} // stan

#endif

