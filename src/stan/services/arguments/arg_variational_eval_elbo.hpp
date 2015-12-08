#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_EVAL_ELBO_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_EVAL_ELBO_HPP

#include <stan/services/arguments/singleton_argument.hpp>

#include <boost/lexical_cast.hpp>
#include <string>

namespace stan {

  namespace services {

    class arg_variational_eval_elbo: public int_argument {
    public:
      arg_variational_eval_elbo(const char *name,
                                const char *desc,
                                double def): int_argument() {
        _name = name;
        _description = desc;
        _validity = "0 < eval_elbo";
        _default = boost::lexical_cast<std::string>(def);
        _default_value = def;
        _constrained = true;
        _good_value = 100.0;
        _bad_value = -1.0;
        _value = _default_value;
      }
      bool is_valid(int value) { return value > 0; }
    };
  }  // services
}  // stan

#endif
