#ifndef STAN__MCMC__BASE_MCMC__HPP
#define STAN__MCMC__BASE_MCMC__HPP

#include <ostream>
#include <string>

#include <stan/mcmc/sample.hpp>

namespace stan {

  namespace mcmc {
    
    class base_mcmc {
      
    public:
      
      base_mcmc(std::ostream* o, std::ostream* e): out_stream_(o), err_stream_(e) {};
      
      virtual ~base_mcmc() {};
      
      virtual sample transition(sample& init_sample) = 0;
      
      std::string name() { return name_; }
      
      virtual void write_sampler_param_names(std::ostream& o) {};
      
      virtual void write_sampler_params(std::ostream& o) {};
      
      virtual void get_sampler_param_names(std::vector<std::string>& names) {};
      
      virtual void get_sampler_params(std::vector<double>& values) {};
      
      virtual void write_sampler_state(std::ostream* o) {};
      
      virtual void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                                std::vector<std::string>& names) {};
      
      virtual void get_sampler_diagnostics(std::vector<double>& values) {};
      
    protected:
      
      std::string name_;
      
      std::ostream* out_stream_;
      std::ostream* err_stream_;
      
    };

  } // mcmc
  
} // stan

#endif

