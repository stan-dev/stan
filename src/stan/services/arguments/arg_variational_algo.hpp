#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_ALGO_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_ALGO_HPP

#include <stan/services/arguments/list_argument.hpp>

#include <stan/services/arguments/arg_variational_meanfield.hpp>
#include <stan/services/arguments/arg_variational_fullrank.hpp>

namespace stan {

  namespace services {

    class arg_variational_algo: public list_argument {
    public:
      arg_variational_algo() {
        _name = "algorithm";
        _description = "Variational inference algorithm";

        _values.push_back(new arg_variational_meanfield());
        _values.push_back(new arg_variational_fullrank());

        _default_cursor = 0;
        _cursor = _default_cursor;
      }
    };
  }  // services
}  // stan

#endif

