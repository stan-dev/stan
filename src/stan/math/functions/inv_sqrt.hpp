#ifndef __STAN__MATH__FUNCTIONS__INV_SQRT_HPP__
#define __STAN__MATH__FUNCTIONS__INV_SQRT_HPP__

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {
    
    template <typename T>
    inline T inv_sqrt(const T x) {
      using std::pow;

      return pow(x, -0.5);
    }

  }
}

#endif
