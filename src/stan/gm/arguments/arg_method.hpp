#ifndef STAN__GM__ARGUMENTS__METHOD__HPP
#define STAN__GM__ARGUMENTS__METHOD__HPP

#include <stan/gm/arguments/list_argument.hpp>

#include <stan/gm/arguments/arg_sample.hpp>
#include <stan/gm/arguments/arg_optimize.hpp>
#include <stan/gm/arguments/arg_diagnose.hpp>
#include <stan/gm/arguments/arg_vbmf.hpp>
#include <stan/gm/arguments/arg_vbfr.hpp>

namespace stan {

  namespace gm {

    class arg_method: public list_argument {

    public:

      arg_method() {

        _name = "method";
        _description = "Analysis method (Note that method= is optional)";

        _values.push_back(new arg_sample());
        _values.push_back(new arg_optimize());
        _values.push_back(new arg_diagnose());
        _values.push_back(new arg_vbmf());
        _values.push_back(new arg_vbfr());

        _default_cursor = 0;
        _cursor = _default_cursor;

      }

    };

  } // gm

} // stan

#endif

