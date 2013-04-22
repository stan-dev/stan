#ifndef __STAN__GM__ARGUMENTS__STEPSIZE__HPP__
#define __STAN__GM__ARGUMENTS__STEPSIZE__HPP__

#include <stan/gm/arguments/sub_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class sarg_stepsize: public sub_argument {
      
    public:
      
      sarg_stepsize(): sub_argument() {
        _name = "stepsize";
      };
      
      bool valid_value(double v) {
        
        if(v > 0) return true;
        
        std::cout << "WARNING: " << v << " is not a valid value for "
                  << _name << "," << std::endl;
        std::cout << "         which must be positive" << std::endl;
        std::cout << "         falling back to default value" << std::endl;
        
        return false;
        
      }
      
      void print_help(std::ostream* s) {
        if(!s) return;
        
        *s << "stepsize - Discretization step size" << std::endl;
        *s << "Valid for stepsize > 0" << std::endl;
        *s << "Defaults to 1" << std::endl;
        
      }
      
    protected:
      
    };
    
  } // gm
  
} // stan

#endif