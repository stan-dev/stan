#ifndef __STAN__MCMC__BASE_MCMC__HPP__
#define __STAN__MCMC__BASE_MCMC__HPP__

#include <fstream>
#include <string>

#include <stan/mcmc/sample.hpp>

namespace stan {

  namespace mcmc {
    
    class base_mcmc {
      
    public:
      
      virtual sample transition(sample& init_sample) = 0;
      
      std::string name() { return _name; }
      
      virtual void write_sampler_param_names(std::ostream& o) {};
      
      virtual void write_sampler_params(std::ostream& o) {};
      
      virtual void get_sampler_param_names(std::vector<std::string>& names) {};
      
      virtual void get_sampler_params(std::vector<double>& values) {};
      
    protected:
      
      std::string _name;
      
    };

  } // mcmc
  
} // stan

#endif

