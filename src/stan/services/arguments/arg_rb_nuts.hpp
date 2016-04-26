#ifndef STAN_SERVICES_ARGUMENTS_ARG_RB_NUTS_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_RB_NUTS_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_max_depth.hpp>

namespace stan {
  namespace services {

    class arg_rb_nuts: public categorical_argument {
    public:
      arg_rb_nuts() {
        _name = "rb_nuts";
        _description = "The Rao-Blackwellized No-U-Turn Sampler";

        _subarguments.push_back(new arg_max_depth());
      }
    };

  }  // services
}  // stan

#endif
