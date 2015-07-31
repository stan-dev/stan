#ifndef STAN_SERVICES_ARGUMENTS_ARG_BFGS_HPP
#define STAN_SERVICES_ARGUMENTS_ARG_BFGS_HPP

#include <stan/services/arguments/categorical_argument.hpp>
#include <stan/services/arguments/arg_init_alpha.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>

namespace stan {
  namespace services {

    class arg_bfgs: public categorical_argument {
    public:
      arg_bfgs() {
        _name = "bfgs";
        _description = "BFGS with linesearch";

        _subarguments.push_back(new arg_init_alpha());
        _subarguments.push_back(
          new arg_tolerance("tol_obj",
                            "Convergence tolerance on absolute changes "
                            "in objective function value",
                            1e-12));
        _subarguments.push_back(
          new arg_tolerance("tol_rel_obj",
                            "Convergence tolerance on relative changes "
                            "in objective function value",
                            1e+4));
        _subarguments.push_back(
          new arg_tolerance("tol_grad",
                            "Convergence tolerance on the norm of the gradient",
                            1e-8));
        _subarguments.push_back(
          new arg_tolerance("tol_rel_grad",
                            "Convergence tolerance on the relative norm "
                            "of the gradient",
                            1e+7));
        _subarguments.push_back(
          new arg_tolerance("tol_param",
                            "Convergence tolerance on changes "
                            "in parameter value",
                            1e-8));
      }
    };

  }  // services
}  // stan

#endif
