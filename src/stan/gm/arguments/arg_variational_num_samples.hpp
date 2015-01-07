#ifndef STAN__GM__ARGUMENTS__VARIATIONAL__NUM__SAMPLES__HPP
#define STAN__GM__ARGUMENTS__VARIATIONAL__NUM__SAMPLES__HPP

#include <stan/gm/arguments/singleton_argument.hpp>

namespace stan {

  namespace gm {

    class arg_variational_num_samples: public int_argument {

    public:

      arg_variational_num_samples(): int_argument() {
        _name = "num_samples";
        _description = "Number of samples for Monte Carlo integrals";
        _validity = "0 <= num_samples";
        _default = "10";
        _default_value = 10;
        _constrained = true;
        _good_value = 2.0;
        _bad_value = -1.0;
        _value = _default_value;
      };

      bool is_valid(int value) { return value >= 0; }

    };

  } // gm

} // stan

#endif
