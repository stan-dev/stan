#ifndef __STAN__MATH__FUNCTIONS__INV_SQUARE_HPP__
#define __STAN__MATH__FUNCTIONS__INV_SQUARE_HPP__

namespace stan {
  namespace math {
    
    template <typename T>
    inline T inv_square(const T x) {
      return 1 / (x * x);
    }

  }
}

#endif
