#ifndef STAN_SERVICES_ARGUMENTS_ARG_OUTPUT_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_OUTPUT_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_output_file.hpp>
#include <stan/services/arguments/arg_diagnostic_file.hpp>
#include <stan/services/arguments/arg_refresh.hpp>

namespace stan {
  namespace services {

    class arg_output: public categorical_argument {
    public:
      arg_output() {
        _name = "output";
        _description = "File output options";

        _subarguments.push_back(new arg_output_file());
        _subarguments.push_back(new arg_diagnostic_file());
        _subarguments.push_back(new arg_refresh());
      }
    };

  }  // services
}  // stan

#endif

