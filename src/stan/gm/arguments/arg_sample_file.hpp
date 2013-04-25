#ifndef __STAN__GM__ARGUMENTS__SAMPLE__FILE__HPP__
#define __STAN__GM__ARGUMENTS__SAMPLE__FILE__HPP__

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_sample_file: public string_argument {
      
    public:
      
      arg_sample_file(): string_argument() {
        _name = "sample";
        _description = "Output file for sample information";
        _default = "samples.csv";
        _default_value = "samples.csv";
        _value = _default_value;
      };
      
    };
    
  } // gm
  
} // stan

#endif