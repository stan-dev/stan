#ifndef STAN__AGRAD__REV__FUNCTIONS__AS_BOOL_HPP
#define STAN__AGRAD__REV__FUNCTIONS__AS_BOOL_HPP

#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {

    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param v Value.
     * @return 1 if argument is equal to zero (or NaN) and 0 otherwise.
     */
    inline int as_bool(const agrad::var& v) {
      return 0.0 != v.vi_->val_;
    }

  }
}
#endif
