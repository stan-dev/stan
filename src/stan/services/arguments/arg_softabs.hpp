#ifndef STAN_SERVICES_ARGUMENTS_ARG_SOFTABS_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_SOFTABS_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_softabs_alpha.hpp>

namespace stan {
  namespace services {

    class arg_softabs: public categorical_argument {
    public:
      arg_softabs() {
        _name = "softabs";
        _description = "Riemannian manifold with SoftAbs metric";

        _subarguments.push_back(new arg_softabs_alpha());
      }
    };

  }  // services
}  // stan

#endif
