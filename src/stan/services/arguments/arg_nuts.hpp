#ifndef STAN_SERVICES_ARGUMENTS_ARG_NUTS_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_NUTS_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>

namespace stan {
  namespace services {

    class arg_nuts: public categorical_argument {
    public:
      arg_nuts() {
        _name = "nuts";
        _description = "The No-U-Turn Sampler";

        _subarguments.push_back(new arg_max_depth());
      }
    };

  }  // services
}  // stan

#endif

