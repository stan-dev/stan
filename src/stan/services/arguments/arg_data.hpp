#ifndef STAN__SERVICES__ARGUMENTS__DATA__HPP
#define STAN__SERVICES__ARGUMENTS__DATA__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_data_file.hpp>

namespace stan {
  
  namespace services {
    
    class arg_data: public categorical_argument {
      
    public:
      
      arg_data(): categorical_argument() {
        
        _name = "data";
        _description = "Input data options";
        
        _subarguments.push_back(new arg_data_file());
        
      };
      
    };
    
  } // services
  
} // stan

#endif
