#ifndef STAN__GM__ARGUMENTS__VARIATIONAL__ALGO__HPP
#define STAN__GM__ARGUMENTS__VARIATIONAL__ALGO__HPP

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_variational_meanfield.hpp>
#include <stan/gm/arguments/arg_variational_fullrank.hpp>

namespace stan {

  namespace gm {

    class arg_variational_algo: public list_argument {

    public:

      arg_variational_algo() {

        _name = "algorithm";
        _description = "Variational inference algorithm";

        _values.push_back(new arg_variational_meanfield());
        _values.push_back(new arg_variational_fullrank());

        _default_cursor = 0;
        _cursor = _default_cursor;

      }

    };

  } // gm

} // stan

#endif

