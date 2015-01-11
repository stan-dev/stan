#ifndef STAN__MCMC__FIXED__PARAM__SAMPLER__HPP
#define STAN__MCMC__FIXED__PARAM__SAMPLER__HPP

#include <iostream>
#include <string>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>

namespace stan {

  namespace mcmc {
    
    class fixed_param_sampler: public base_mcmc {
      
    public:
      
      fixed_param_sampler(std::ostream* o = &std::cout, std::ostream* e = 0):
        base_mcmc(o, e) { this->name_ = "Fixed Parameter Sampler"; }
      
      sample transition(sample& init_sample) { return init_sample; }

      
    };

  } // mcmc
  
} // stan

#endif

