#ifndef __STAN__DIFF__REV__OPERATOR_PLUS_EQUAL_HPP__
#define __STAN__DIFF__REV__OPERATOR_PLUS_EQUAL_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/operator_addition.hpp>

namespace stan {
  namespace diff {

    inline var& var::operator+=(const var& b) {
      vi_ = new add_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator+=(const double b) {
      if (b == 0.0)
        return *this;
      vi_ = new add_vd_vari(vi_,b);
      return *this;
    }

  }
}
#endif
