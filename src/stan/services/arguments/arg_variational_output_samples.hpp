#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_OUTPUT_SAMPLES_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_OUTPUT_SAMPLES_HPP

#include <stan/services/arguments/singleton_argument.hpp>

#include <boost/lexical_cast.hpp>
#include <string>

namespace stan {

  namespace services {

    class arg_variational_output_samples: public int_argument {
    public:
      arg_variational_output_samples(const char *name,
                                     const char *desc,
                                     double def): int_argument() {
        _name = name;
        _description = desc;
        _validity = "0 < output_samples";
        _default = boost::lexical_cast<std::string>(def);
        _default_value = def;
        _constrained = true;
        _good_value = 1000.0;
        _bad_value = -1.0;
        _value = _default_value;
      }
      bool is_valid(int value) { return value > 0; }
    };
  }  // services
}  // stan

#endif
