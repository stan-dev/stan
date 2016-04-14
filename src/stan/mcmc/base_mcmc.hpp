#ifndef STAN_MCMC_BASE_MCMC_HPP
#define STAN_MCMC_BASE_MCMC_HPP

#include <stan/interface_callbacks/writer/base_writer.hpp>
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

      virtual sample
      transition(sample& init_sample,
                 interface_callbacks::writer::base_writer& info_writer,
                 interface_callbacks::writer::base_writer& error_writer) = 0;

      virtual void get_sampler_param_names(std::vector<std::string>& names) {}

      virtual void get_sampler_params(std::vector<double>& values) {}

      virtual void
      write_sampler_state(interface_callbacks::writer::base_writer& writer) {}

      virtual void
      get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                   std::vector<std::string>& names) {}

      virtual void get_sampler_diagnostics(std::vector<double>& values) {}
    };

  }  // mcmc
}  // stan
#endif
