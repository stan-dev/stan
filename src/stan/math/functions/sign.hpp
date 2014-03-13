#ifndef __STAN__MATH__FUNCTIONS__SIGN_HPP__
#define __STAN__MATH__FUNCTIONS__SIGN_HPP__

namespace stan {
  namespace math {
    template<typename T>
    inline int sign(const T& z) {
      return (z == 0) ? 0 : z < 0 ? -1 : 1;
    }
  }
}

#endif

