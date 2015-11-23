#ifndef STAN_SERVICES_ARGUMENTS_VARIATIONAL_HPP
#define STAN_SERVICES_ARGUMENTS_VARIATIONAL_HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_variational_algo.hpp>
#include <stan/services/arguments/arg_variational_iter.hpp>
#include <stan/services/arguments/arg_variational_num_samples.hpp>
#include <stan/services/arguments/arg_variational_eta.hpp>
#include <stan/services/arguments/arg_variational_adapt.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>
#include <stan/services/arguments/arg_variational_eval_elbo.hpp>
#include <stan/services/arguments/arg_variational_output_samples.hpp>

namespace stan {
  namespace services {

    class arg_variational: public categorical_argument {
    public:
      arg_variational() {
        _name = "variational";
        _description = "Variational inference";

        _subarguments.push_back(new arg_variational_algo());
        _subarguments.push_back(new arg_variational_iter());
        _subarguments.push_back(new arg_variational_num_samples("grad_samples",
          "Number of samples for Monte Carlo estimate of gradients", 1));
        _subarguments.push_back(new arg_variational_num_samples
                                ("elbo_samples",
                                 "Number of samples for Monte Carlo estimate "
                                 "of ELBO (objective function)",
                                 100));
        _subarguments.push_back(new arg_variational_eta());
        _subarguments.push_back(new arg_variational_adapt());
        _subarguments.push_back(new arg_tolerance("tol_rel_obj",
          "Convergence tolerance on the relative norm of the objective", 1e-2));
        _subarguments.push_back(new arg_variational_eval_elbo("eval_elbo",
          "Evaluate ELBO every Nth iteration", 100));
        _subarguments.push_back(new arg_variational_output_samples
                                ("output_samples",
                                 "Number of posterior samples to draw and save",
                                 1000));
      }
    };

  }  // services
}  // stan

#endif

