#ifndef STAN__SERVICES__ARGUMENTS__ARGUMENT__PROBE__HPP
#define STAN__SERVICES__ARGUMENTS__ARGUMENT__PROBE__HPP

#include <string>
#include <vector>

#include <stan/services/arguments/argument.hpp>

namespace stan {
  
  namespace services {
    
    class argument_probe {
      
    public:
      
      argument_probe(std::vector<argument*>& valid_args)
        : _arguments(valid_args) {}
      
      template <class Writer>
      void probe_args(Writer& writer) {

        for (std::vector<argument*>::iterator arg_it = _arguments.begin();
             arg_it != _arguments.end(); ++arg_it)
          (*arg_it)->probe_args(*arg_it, writer);
          
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      
    };
    
  } // services
  
} // stan

#endif

