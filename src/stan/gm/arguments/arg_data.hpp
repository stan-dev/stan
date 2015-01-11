#ifndef STAN__GM__ARGUMENTS__DATA__HPP
#define STAN__GM__ARGUMENTS__DATA__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_data_file.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_data: public categorical_argument {
      
    public:
      
      arg_data(): categorical_argument() {
        
        _name = "data";
        _description = "Input data options";
        
        _subarguments.push_back(new arg_data_file());
        
      };
      
    };
    
  } // gm
  
} // stan

#endif
