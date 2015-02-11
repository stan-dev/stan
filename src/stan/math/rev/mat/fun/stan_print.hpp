#ifndef STAN__AGRAD__REV__MATRIX__STAN_PRINT_HPP
#define STAN__AGRAD__REV__MATRIX__STAN_PRINT_HPP

#include <ostream>
#include <stan/math/rev/arr/meta/var.hpp>

namespace stan {
  namespace agrad {

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}
#endif
