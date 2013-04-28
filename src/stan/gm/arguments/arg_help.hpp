#ifndef __STAN__GM__ARGUMENTS__HELP__HPP__
#define __STAN__GM__ARGUMENTS__HELP__HPP__

#include <stan/gm/arguments/unvalued_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_help: public unvalued_argument {
      
    public:
      
      arg_help() {
        
        _name = "help";
        _description = "Display valid arguments";
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

