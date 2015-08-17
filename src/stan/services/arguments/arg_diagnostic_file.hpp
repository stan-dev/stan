#ifndef STAN_SERVICES_ARGUMENTS_ARG_DIAGNOSTIC_FILE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_DIAGNOSTIC_FILE_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_diagnostic_file: public string_argument {
    public:
      arg_diagnostic_file(): string_argument() {
        _name = "diagnostic_file";
        _description = "Auxiliary output file for diagnostic information";
        _validity = "Path to existing file";
        _default = "\"\"";
        _default_value = "";
        _constrained = false;
        _good_value = "good";
        _value = _default_value;
      }
    };

  }  // services
}  // stan

#endif
