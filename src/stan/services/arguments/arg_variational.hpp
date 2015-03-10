#ifndef STAN__SERVICES__ARGUMENTS__VARIATIONAL__HPP
#define STAN__SERVICES__ARGUMENTS__VARIATIONAL__HPP

#include <stan/services/arguments/categorical_argument.hpp>

#include <stan/services/arguments/arg_variational_algo.hpp>
#include <stan/services/arguments/arg_variational_num_samples.hpp>
#include <stan/services/arguments/arg_iter.hpp>
#include <stan/services/arguments/arg_tolerance.hpp>
#include <stan/services/arguments/arg_save_variational.hpp>

namespace stan {

  namespace services {

    class arg_variational: public categorical_argument {

    public:

      arg_variational() {

        _name = "variational";
        _description = "Variational inference";

        _subarguments.push_back(new arg_variational_algo());
        _subarguments.push_back(new arg_variational_num_samples());
        _subarguments.push_back(new arg_iter());
        _subarguments.push_back(new arg_tolerance("tol_rel_param",
          "Convergence tolerance on the relative norm of the parameters",1e+7));
        _subarguments.push_back(new arg_save_variational());

      }

    };

  } // services

} // stan

#endif

