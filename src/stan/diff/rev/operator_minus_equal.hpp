#ifndef __STAN__DIFF__REV__OPERATOR_MINUS_EQUAL_HPP__
#define __STAN__DIFF__REV__OPERATOR_MINUS_EQUAL_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/operator_subtraction.hpp>

namespace stan {
  namespace diff {

    inline var& var::operator-=(const var& b) {
      vi_ = new subtract_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator-=(const double b) {
      if (b == 0.0)
        return *this;
      vi_ = new subtract_vd_vari(vi_,b);
      return *this;
    }

  }
}
#endif
