#ifndef STAN__SERVICES__ARGUMENTS__ARGUMENT__PROBE__HPP
#define STAN__SERVICES__ARGUMENTS__ARGUMENT__PROBE__HPP

#include <string>
#include <vector>
#include <sstream>

#include <stan/interface_callbacks/writer/base_writer.hpp>

#include <stan/services/arguments/argument.hpp>

namespace stan {
  
  namespace services {
    
    class argument_probe {
      
    public:
      
      argument_probe(std::vector<argument*>& valid_args)
        : _arguments(valid_args) {}
      
      void probe_args(interface_callbacks::writer::base_writer&) {

        for (std::vector<argument*>::iterator arg_it = _arguments.begin();
             arg_it != _arguments.end(); ++arg_it)
          (*arg_it)->probe_args(*arg_it, w);
          
      }
      
    protected:
      
      std::vector<argument*>& _arguments;
      
    };
    
  } // services
  
} // stan

#endif

