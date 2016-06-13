#ifndef STAN_OLD_SERVICES_ARGUMENTS_ARG_RANDOM_HPP
#define STAN_OLD_SERVICES_ARGUMENTS_ARG_RANDOM_HPP

#include <stan/old_services/arguments/categorical_argument.hpp>
#include <stan/old_services/arguments/arg_seed.hpp>

namespace stan {
  namespace services {

    class arg_random: public categorical_argument {
    public:
      arg_random() {
        _name = "random";
        _description = "Random number configuration";

        _subarguments.push_back(new arg_seed());
      }
    };

  }  // services
}  // stan

#endif

