#ifndef STAN__MATH__FUNCTIONS__SIGN_HPP
#define STAN__MATH__FUNCTIONS__SIGN_HPP

namespace stan {
  namespace math {

    // returns 1 if NaN is passed in.
    template<typename T>
    inline int sign(const T& z) {
      return (z == 0) ? 0 : z < 0 ? -1 : 1;
    }
  }
}

#endif

