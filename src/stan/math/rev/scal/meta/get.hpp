#ifndef STAN__MATH__REV__SCAL__META__GET_HPP
#define STAN__MATH__REV__SCAL__META__GET_HPP

#include <stan/math/rev/core.hpp>

namespace stan {

  inline const stan::agrad::var &get(const stan::agrad::var& x, size_t n) {
    return x;
  }

}
#endif

