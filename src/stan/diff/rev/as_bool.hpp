#ifndef __STAN__DIFF__REV__AS_BOOL_HPP__
#define __STAN__DIFF__REV__AS_BOOL_HPP__

#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {

    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param x Value.
     * @return 1 if argument is equal to zero and 0 otherwise.
     */
    inline int as_bool(const diff::var& v) {
      return 0.0 != v.vi_->val_;
    }

  }
}
#endif
