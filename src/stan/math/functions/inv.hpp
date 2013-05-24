#ifndef __STAN__MATH__FUNCTIONS__INV_HPP__
#define __STAN__MATH__FUNCTIONS__INV_HPP__

namespace stan {
  namespace math {
    
    template <typename T>
    inline T inv(const T x) {
      return 1 / x;
    }

  }
}

#endif
