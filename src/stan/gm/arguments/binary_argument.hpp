#ifndef __STAN__GM__ARGUMENTS__BINARY__ARGUMENT__BETA__
#define __STAN__GM__ARGUMENTS__BINARY__ARGUMENT__BETA__

#include <stan/gm/arguments/list_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class binary_argument: public list_argument {
      
    public:
      
      binary_argument() {
        _values.push_back("true");
        _values.push_back("false");
      }
      
    };
    
  } // gm
  
} // stan

#endif