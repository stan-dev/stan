#ifndef __STAN__GM__ARGUMENTS__HMC__HPP__
#define __STAN__GM__ARGUMENTS__HMC__HPP__

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_engine.hpp>
#include <stan/gm/arguments/arg_metric.hpp>
#include <stan/gm/arguments/arg_stepsize.hpp>
#include <stan/gm/arguments/arg_stepsize_jitter.hpp>

namespace stan {
  
  namespace gm {
    
    class arg_hmc: public categorical_argument {
      
    public:
      
      arg_hmc() {
        
        _name = "hmc";
        _description = "Hamiltonian Monte Carlo";
        
        _subarguments.push_back(new arg_engine());
        _subarguments.push_back(new arg_metric());
        _subarguments.push_back(new arg_stepsize());
        _subarguments.push_back(new arg_stepsize_jitter());
        
      }
      
    };
    
  } // gm
  
} // stan

#endif

