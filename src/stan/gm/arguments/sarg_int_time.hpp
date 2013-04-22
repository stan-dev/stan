#ifndef __STAN__GM__ARGUMENTS__INT_TIME__HPP__
#define __STAN__GM__ARGUMENTS__INT_TIME__HPP__

#include <stan/gm/arguments/sub_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class sarg_int_time: public sub_argument {
      
    public:
      
      sarg_int_time(): sub_argument() {
        _name = "int_time";
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
        
        *s << "int_time - Total integration time" << std::endl;
        *s << "Valid for int_time > 0" << std::endl;
        *s << "Defaults to 2 * pi" << std::endl;
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

