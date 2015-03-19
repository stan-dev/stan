#ifndef STAN__SERVICES__ARGUMENTS__RANDOM__HPP
#define STAN__SERVICES__ARGUMENTS__RANDOM__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_seed.hpp>

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

  } // services

} // stan

#endif

