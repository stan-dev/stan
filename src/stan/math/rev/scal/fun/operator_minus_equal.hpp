#ifndef STAN__MATH__REV__SCAL__FUN__OPERATOR_MINUS_EQUAL_HPP
#define STAN__MATH__REV__SCAL__FUN__OPERATOR_MINUS_EQUAL_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/scal/fun/operator_subtraction.hpp>

namespace stan {
  namespace agrad {

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
