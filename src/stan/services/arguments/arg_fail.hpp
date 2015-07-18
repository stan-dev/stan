#ifndef STAN_SERVICES_ARGUMENTS_ARG_FAIL_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_FAIL_HPP

#include <stan/services/arguments/unvalued_argument.hpp>

namespace stan {
  namespace services {

    class arg_fail: public unvalued_argument {
    public:
      arg_fail() {
        _name = "fail";
        _description = "Dummy argument to induce failures for testing";
      }
    };

  }  // services
}  // stan

#endif

