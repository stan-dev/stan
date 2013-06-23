#ifndef __STAN__MCMC__BASE_MCMC__HPP__
#define __STAN__MCMC__BASE_MCMC__HPP__

#include <ostream>
#include <string>

#include <stan/mcmc/sample.hpp>

namespace stan {

  namespace mcmc {
    
    class base_mcmc {
      
    public:
      
      base_mcmc(std::ostream* o, std::ostream* e): _out_stream(o), _err_stream(e) {};
      
      virtual ~base_mcmc() {};
      
      virtual sample transition(sample& init_sample) = 0;
      
      std::string name() { return _name; }
      
      virtual void write_sampler_param_names(std::ostream& o) {};
      
      virtual void write_sampler_params(std::ostream& o) {};
      
      virtual void get_sampler_param_names(std::vector<std::string>& names) {};
      
      virtual void get_sampler_params(std::vector<double>& values) {};
      
      virtual void write_sampler_state(std::ostream* o) {};
      
      virtual void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                                std::vector<std::string>& names) {};
      
      virtual void get_sampler_diagnostics(std::vector<double>& values) {};
      
    protected:
      
      std::string _name;
      
      std::ostream* _out_stream;
      std::ostream* _err_stream;
      
    };

  } // mcmc
  
} // stan

#endif

