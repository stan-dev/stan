#ifndef STAN_MCMC_BASE_MCMC_HPP
#define STAN_MCMC_BASE_MCMC_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
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
  base_mcmc(const base_mcmc& other) = default;
  base_mcmc(base_mcmc&& other) = default;
  base_mcmc& operator=(const base_mcmc&) = default;
  base_mcmc& operator=(base_mcmc&&) = default;
  inline virtual sample transition(sample& init_sample,
                                   callbacks::logger& logger)
      = 0;

  inline virtual void get_sampler_param_names(std::vector<std::string>& names) {
  }

  inline virtual void get_sampler_params(std::vector<double>& values) {}

  inline virtual void write_sampler_state(callbacks::writer& writer) {}

  inline virtual void get_sampler_diagnostic_names(
      std::vector<std::string>& model_names, std::vector<std::string>& names) {}

  inline virtual void get_sampler_diagnostics(std::vector<double>& values) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
