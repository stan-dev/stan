#ifndef __STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP__
#define __STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP__

#include <string>
#include <vector>
#include <fstream>

#include <stan/gm/arguments/argument.hpp>
#include <stan/gm/arguments/arg_method.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_probe {
      
    public:
      
      argument_probe(std::vector<argument*>& valid_args, std::string dir= "")
        : _arguments(valid_args),
          _dir(dir) {
        _arguments.insert(_arguments.begin(), new arg_method());
      }
      
      void probe_args() {

        for (std::vector<argument*>::iterator arg_it = _arguments.begin();
             arg_it != _arguments.end(); ++arg_it)
          (*arg_it)->probe_args(*arg_it, _dir);
          
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      std::string _dir;
      
    };
    
  } // gm
  
} // stan

#endif

