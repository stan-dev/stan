#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP__

#include <string>
#include <vector>
#include <sstream>

#include <stan/gm/arguments/argument.hpp>
#include <stan/gm/arguments/arg_method.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_probe {
      
    public:
      
      argument_probe(std::vector<argument*>& valid_args)
        : _arguments(valid_args) {
        _arguments.insert(_arguments.begin(), new arg_method());
      }
      
      void probe_args(std::stringstream& s) {

        for (std::vector<argument*>::iterator arg_it = _arguments.begin();
             arg_it != _arguments.end(); ++arg_it)
          (*arg_it)->probe_args(*arg_it, s);
          
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      
    };
    
  } // gm
  
} // stan

#endif

