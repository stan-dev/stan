#ifndef STAN_SERVICES_ARGUMENTS_ARG_VARIATIONAL_ADAPT_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_VARIATIONAL_ADAPT_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_variational_adapt_engaged.hpp>
#include <stan/services/arguments/arg_variational_adapt_iter.hpp>

namespace stan {
  namespace services {

    class arg_variational_adapt: public categorical_argument {
    public:
      arg_variational_adapt() {
        _name = "adapt";
        _description = "Eta Adaptation for Variational Inference";

        _subarguments.push_back(new arg_variational_adapt_engaged());
        _subarguments.push_back(new arg_variational_adapt_iter());
      }
    };

  }  // services
}  // stan
#endif

