#ifndef STAN__MATH__REV__CORE__OPERATOR_MULTIPLY_EQUAL_HPP
#define STAN__MATH__REV__CORE__OPERATOR_MULTIPLY_EQUAL_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {
  namespace agrad {

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
