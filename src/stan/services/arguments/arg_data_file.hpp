#ifndef STAN_SERVICES_ARGUMENTS_ARG_DATA_FILE_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_DATA_FILE_HPP

#include <stan/services/arguments/singleton_argument.hpp>

namespace stan {
  namespace services {

    class arg_data_file: public string_argument {
    public:
      arg_data_file(): string_argument() {
        _name = "file";
        _description = "Input data file";
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
