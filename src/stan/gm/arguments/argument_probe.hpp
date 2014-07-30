#ifndef STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP
#define STAN__GM__ARGUMENTS__ARGUMENT__PROBE__HPP

#include <string>
#include <vector>
#include <sstream>

#include <stan/gm/arguments/argument.hpp>

namespace stan {
  
  namespace gm {
    
    class argument_probe {
      
    public:
      
      argument_probe(std::vector<argument*>& valid_args)
        : _arguments(valid_args) {}
      
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

