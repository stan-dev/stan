#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_ADAPT_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_ADAPT_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>
#include <stan/old_services/arguments/arg_adapt_engaged.hpp>
#include <stan/old_services/arguments/arg_adapt_gamma.hpp>
#include <stan/old_services/arguments/arg_adapt_delta.hpp>
#include <stan/old_services/arguments/arg_adapt_kappa.hpp>
#include <stan/old_services/arguments/arg_adapt_t0.hpp>
#include <stan/old_services/arguments/arg_adapt_init_buffer.hpp>
#include <stan/old_services/arguments/arg_adapt_term_buffer.hpp>
#include <stan/old_services/arguments/arg_adapt_window.hpp>

namespace stan {
  namespace services {
    class arg_adapt: public categorical_argument {
    public:
      arg_adapt() {
        _name = "adapt";
        _description = "Warmup Adaptation";

        _subarguments.push_back(new arg_adapt_engaged());
        _subarguments.push_back(new arg_adapt_gamma());
        _subarguments.push_back(new arg_adapt_delta());
        _subarguments.push_back(new arg_adapt_kappa());
        _subarguments.push_back(new arg_adapt_t0());
        _subarguments.push_back(new arg_adapt_init_buffer());
        _subarguments.push_back(new arg_adapt_term_buffer());
        _subarguments.push_back(new arg_adapt_window());
      }
    };

  }  // services
}  // stan
#endif

