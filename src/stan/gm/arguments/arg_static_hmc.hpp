#ifndef __STAN__GM__ARGUMENTS__STATIC__HMC__HPP__
#define __STAN__GM__ARGUMENTS__STATIC__HMC__HPP__

#include <stan/gm/arguments/argument.hpp>

#include <stan/gm/arguments/sarg_int_time.hpp>
#include <stan/gm/arguments/sarg_stepsize.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_static_hmc: public argument {
      
    public:
      
      arg_static_hmc() {
        
        _name = "static_hmc";
        
        _valid_subarguments.clear();
        _valid_subarguments.push_back(new sarg_int_time());
        _valid_subarguments.push_back(new sarg_stepsize());
        
      }
      
      void print_help(std::ostream* s) {
        if(!s) return;
        
        *s << "static_hmc - Hamiltonian Monte Carlo with a static integration time" << std::endl;
        *s << "Valid subarguments:" << std::endl;
        
        for (std::vector<sub_argument*>::iterator it = _valid_subarguments.begin();
             it != _valid_subarguments.end(); ++it) {
          (*it)->print_help(s);
        }
        
      }
 
    };
    
  } // gm
  
} // stan

#endif

