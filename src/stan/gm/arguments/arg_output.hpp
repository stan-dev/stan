#ifndef __STAN__GM__ARGUMENTS__OUTPUT__HPP__
#define __STAN__GM__ARGUMENTS__OUTPUT__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_sample_file.hpp>
#include <stan/gm/arguments/arg_append_sample.hpp>
#include <stan/gm/arguments/arg_diagnostic_file.hpp>
#include <stan/gm/arguments/arg_append_diagnostic.hpp>
#include <stan/gm/arguments/arg_refresh.hpp>
#include <stan/gm/arguments/arg_save_warmup.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_output: public categorical_argument {
      
    public:
      
      arg_output() {
        
        _name = "output";
        _description = "File output options";
        
        _subarguments.push_back(new arg_sample_file());
        _subarguments.push_back(new arg_append_sample());
        _subarguments.push_back(new arg_diagnostic_file());
        _subarguments.push_back(new arg_append_diagnostic());
        _subarguments.push_back(new arg_refresh());
        _subarguments.push_back(new arg_save_warmup());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

