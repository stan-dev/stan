#ifndef STAN__MATH__REV__CORE__OPERATOR_PLUS_EQUAL_HPP
#define STAN__MATH__REV__CORE__OPERATOR_PLUS_EQUAL_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/core/operator_addition.hpp>

namespace stan {
  namespace agrad {

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
