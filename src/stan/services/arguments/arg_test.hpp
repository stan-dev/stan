#ifndef STAN__SERVICES__ARGUMENTS__TEST__HPP
#define STAN__SERVICES__ARGUMENTS__TEST__HPP

#include <stan/services/arguments/list_argument.hpp>

#include <stan/services/arguments/arg_test_gradient.hpp>

namespace stan {

  namespace services {

    class arg_test: public list_argument {

    public:

      arg_test() {

        _name = "test";
        _description = "Diagnostic test";

        _values.push_back(new arg_test_gradient());

        _default_cursor = 0;
        _cursor = _default_cursor;

      }

    };

  } // services

} // stan

#endif

