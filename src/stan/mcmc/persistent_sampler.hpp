#ifndef __STAN__MCMC__PERSISTENT__SAMPLER__HPP__
#define __STAN__MCMC__PERSISTENT__SAMPLER__HPP__

#include <iostream>
#include <string>

#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>

namespace stan {

  namespace mcmc {
    
    class persistent_sampler: public base_mcmc {
      
    public:
      
      persistent_sampler(std::ostream* o = &std::cout, std::ostream* e = 0):
        base_mcmc(o, e) { this->_name = "Persistent Sampler"; }
      
      sample transition(sample& init_sample) { return init_sample; }
      
      std::string name() { return _name; }

      
    };

  } // mcmc
  
} // stan

#endif

