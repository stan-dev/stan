#ifndef STAN_MCMC_FIXED_PARAM_SAMPLER_HPP
#define STAN_MCMC_FIXED_PARAM_SAMPLER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_mcmc.hpp>
#include <stan/mcmc/sample.hpp>

namespace stan {
namespace mcmc {

class fixed_param_sampler : public base_mcmc<fixed_param_sampler> {
 public:

  inline void get_sampler_param_names(std::vector<std::string>& names) {}

  inline void get_sampler_params(std::vector<double>& values) {}

  inline void write_sampler_state(callbacks::writer& writer) {
  }

  inline void get_sampler_diagnostic_names(
      std::vector<std::string>& model_names, std::vector<std::string>& names) {
  }

  inline void get_sampler_diagnostics(std::vector<double>& values) {
  }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    return init_sample;
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
