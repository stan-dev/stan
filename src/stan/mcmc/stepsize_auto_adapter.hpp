#ifndef STAN_MCMC_STEPSIZE_AUTO_ADAPTER_HPP
#define STAN_MCMC_STEPSIZE_AUTO_ADAPTER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/auto_adaptation.hpp>

namespace stan {

namespace mcmc {

class stepsize_auto_adapter : public base_adapter {
 public:
  explicit stepsize_auto_adapter(int n) : auto_adaptation_(n) {}

  stepsize_adaptation& get_stepsize_adaptation() {
    return stepsize_adaptation_;
  }

  auto_adaptation& get_auto_adaptation() { return auto_adaptation_; }

  void set_window_params(unsigned int num_warmup, unsigned int init_buffer,
                         unsigned int term_buffer, unsigned int base_window,
                         callbacks::logger& logger) {
    auto_adaptation_.set_window_params(num_warmup, init_buffer, term_buffer,
                                       base_window, logger);
  }

 protected:
  stepsize_adaptation stepsize_adaptation_;
  auto_adaptation auto_adaptation_;
};

}  // namespace mcmc

}  // namespace stan

#endif
