#ifndef STAN_MCMC_BASE_MCMC_HPP
#define STAN_MCMC_BASE_MCMC_HPP

#include <stan/mcmc/sample.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace stan {

  namespace mcmc {

    class base_mcmc {
    public:
      base_mcmc() {}

      virtual ~base_mcmc() {}

      virtual sample transition(sample& init_sample) = 0;

      std::string name() {
        return name_;
      }

      virtual void get_sampler_param_names(std::vector<std::string>& names) {}
      virtual void get_sampler_params(std::vector<double>& values) {}

      virtual void get_sampler_diagnostic_names
        (std::vector<std::string>& model_names,
         std::vector<std::string>& names) {}
      virtual void get_sampler_diagnostics(std::vector<double>& values) {}

      std::string flush_info_buffer() { return std::string(); }
      std::string flush_err_buffer() { return std::string(); }

      void clear_buffers() {}

    protected:
      std::string name_;
    };

  }  // mcmc

}  // stan

#endif

