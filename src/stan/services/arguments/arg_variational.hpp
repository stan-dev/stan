#ifndef STAN__SERVICES__ARGUMENTS__VARIATIONAL__HPP
#define STAN__SERVICES__ARGUMENTS__VARIATIONAL__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_variational_algo.hpp>
#include <stan/services/arguments/arg_variational_num_samples.hpp>
#include <stan/services/arguments/arg_variational_eta_stepsize.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>

namespace stan {

  namespace services {

    class arg_variational: public categorical_argument {

    public:

      arg_variational() {

        _name = "variational";
        _description = "Variational inference";

        _subarguments.push_back(new arg_variational_algo());
        _subarguments.push_back(new arg_variational_num_samples("grad_samples",
          "Number of samples for Monte Carlo estimate of gradients", 10));
        _subarguments.push_back(new arg_variational_num_samples("elbo_samples",
          "Number of samples for Monte Carlo estimate of ELBO (objective function)", 100));
        _subarguments.push_back(new arg_variational_eta_stepsize());
        _subarguments.push_back(new arg_iter());
        _subarguments.push_back(new arg_tolerance("tol_rel_obj",
          "Convergence tolerance on the relative norm of the objective", 1e-2));

      }

    };

  } // services

} // stan

#endif

