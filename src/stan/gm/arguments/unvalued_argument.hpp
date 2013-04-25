#ifndef __STAN__GM__ARGUMENTS__UNVALUED__ARGUMENT__BETA__
#define __STAN__GM__ARGUMENTS__UNVALUED__ARGUMENT__BETA__

#include <vector>
#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class unvalued_argument: public argument {
      
    public:
      
      void print(std::ostream* s, int depth) {};
      void print_help(std::ostream* s, int depth) {};
      
    };
    
  } // gm
  
} // stan

#endif