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

template <typename Derived>
class base_mcmc {
 public:
  // mutator for the derived type
  inline Derived& derived() { return static_cast<Derived&>(*this); }
  // inspector to the derived class
  inline const Derived& derived() const {
    return static_cast<Derived const&>(*this);
  }

  inline sample transition(sample& init_sample, callbacks::logger& logger) {
    return this->derived().transition(init_sample, logger);
  };

  inline void get_sampler_param_names(std::vector<std::string>& names) {
    this->derived().get_sampler_param_names(names);
  }

  inline void get_sampler_params(std::vector<double>& values) {
    return this->derived().get_sampler_params(values);
  }

  inline void write_sampler_state(callbacks::writer& writer) {
    return this->derived().write_sampler_state(writer);
  }

  inline void get_sampler_diagnostic_names(
      std::vector<std::string>& model_names, std::vector<std::string>& names) {
    return this->derived().get_sampler_diagnostic_names(model_names, names);
  }

  inline void get_sampler_diagnostics(std::vector<double>& values) {
    return this->derived().get_sampler_diagnostics(values);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
