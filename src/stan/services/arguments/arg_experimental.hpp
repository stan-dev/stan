#ifndef STAN__SERVICES__ARGUMENTS__EXPERIMENTAL__HPP
#define STAN__SERVICES__ARGUMENTS__EXPERIMENTAL__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_variational.hpp>

namespace stan {

  namespace services {

    class arg_experimental: public categorical_argument {
    public:
      arg_experimental() {
        _name = "experimental";
        _description = "Experimental Algorithms";

        _subarguments.push_back(new arg_variational());
      }
    };
  }  // services
}  // stan

#endif

