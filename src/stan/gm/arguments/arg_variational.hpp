#ifndef STAN__GM__ARGUMENTS__VARIATIONAL__HPP
#define STAN__GM__ARGUMENTS__VARIATIONAL__HPP

#include <stan/gm/arguments/categorical_argument.hpp>

#include <stan/gm/arguments/arg_variational_algo.hpp>
#include <stan/gm/arguments/arg_variational_num_samples.hpp>
#include <stan/gm/arguments/arg_iter.hpp>
#include <stan/gm/arguments/arg_tolerance.hpp>
#include <stan/gm/arguments/arg_save_variational.hpp>

namespace stan {

  namespace gm {

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

  } // gm

} // stan

#endif

