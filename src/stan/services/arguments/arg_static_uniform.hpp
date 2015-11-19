#ifndef STAN_SERVICES_ARGUMENTS_ARG_STATIC_UNIFORM_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_STATIC_UNIFORM_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_num_leapfrog.hpp>

namespace stan {
  namespace services {

    class arg_static_uniform: public categorical_argument {
    public:
      arg_static_uniform() {
        _name = "static_uniform";
        _description = "Uniform Sample from Static Trajectory";

        _subarguments.push_back(new arg_num_leapfrog());
      }
    };

  }  // services
}  // stan

#endif

