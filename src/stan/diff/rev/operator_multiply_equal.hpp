#ifndef __STAN__DIFF__REV__OPERATOR_MULTIPLY_EQUAL_HPP__
#define __STAN__DIFF__REV__OPERATOR_MULTIPLY_EQUAL_HPP__

#include <stan/diff/rev/var.hpp>
#include <stan/diff/rev/operator_multiplication.hpp>

namespace stan {
  namespace diff {

    inline var& var::operator*=(const var& b) {
      vi_ = new multiply_vv_vari(vi_,b.vi_);
      return *this;
    }

    inline var& var::operator*=(const double b) {
      if (b == 1.0)
        return *this;
      vi_ = new multiply_vd_vari(vi_,b);
      return *this;
    }

  }
}
#endif
