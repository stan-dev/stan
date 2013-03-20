#ifndef __STAN__MATH__MATRIX__PROD_HPP__
#define __STAN__MATH__MATRIX__PROD_HPP__

#include <stan/math/matrix.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the product of the coefficients of the specified
     * column vector.
     * @param v Specified vector.
     * @return Product of coefficients of vector.
     */
    template <typename T, int R, int C>
    inline T prod(const Eigen::Matrix<T,R,C>& v) {
      if (v.size() == 0) return 1.0;
      return v.prod();
    }
    
  }
}
#endif
