#ifndef STAN_SERVICES_ARGUMENTS_ARG_FIXED_PARAM_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_FIXED_PARAM_HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  namespace services {

    class arg_fixed_param: public unvalued_argument {
    public:
      arg_fixed_param() {
        _name = "fixed_param";
        _description = "Fixed Parameter Sampler";
      }
    };

  }  // services
}  // stan

#endif

