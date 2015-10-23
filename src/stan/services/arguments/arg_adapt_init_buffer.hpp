#ifndef STAN_SERVICES_ARGUMENTS_ARG_ADAPT_INIT_BUFFER_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_ADAPT_INIT_BUFFER_HPP

#include <stan/services/arguments/singleton_argument.hpp>
#include <string>

namespace stan {
  namespace services {

    class arg_adapt_init_buffer: public u_int_argument {
    public:
      arg_adapt_init_buffer(): u_int_argument() {
        _name = "init_buffer";
        _description = std::string("Width of initial fast adaptation interval");
        _default = "75";
        _default_value = 75;
        _value = _default_value;
      }
    };

  }  // services
}  // stan

#endif

