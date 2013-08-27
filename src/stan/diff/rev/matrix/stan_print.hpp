#ifndef __STAN__DIFF__REV__MATRIX__STAN_PRINT_HPP__
#define __STAN__DIFF__REV__MATRIX__STAN_PRINT_HPP__

#include <ostream>
#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {

    void stan_print(std::ostream* o, const var& x) {
      *o << x.val();
    }

  }
}
#endif
