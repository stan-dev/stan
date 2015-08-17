#ifndef STAN_SERVICES_ARGUMENTS_ARG_METRIC_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_METRIC_HPP

#include <stan/services/arguments/list_argument.hpp>
#include <stan/services/arguments/arg_unit_e.hpp>
#include <stan/services/arguments/arg_diag_e.hpp>
#include <stan/services/arguments/arg_dense_e.hpp>

namespace stan {
  namespace services {

    class arg_metric: public list_argument {
    public:
      arg_metric() {
        _name = "metric";
        _description = "Geometry of base manifold";

        _values.push_back(new arg_unit_e());
        _values.push_back(new arg_diag_e());
        _values.push_back(new arg_dense_e());

        _default_cursor = 1;
        _cursor = _default_cursor;
      }
    };

  }  // services
}  // stan

#endif

