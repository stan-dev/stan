#ifndef STAN__MATH__MATRIX__PROD_HPP
#define STAN__MATH__MATRIX__PROD_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the product of the coefficients of the specified
     * standard vector.
     * @param v Specified vector.
     * @return Product of coefficients of vector.
     */
    template <typename T>
    inline T prod(const std::vector<T>& v) {
      if (v.size() == 0) return 1;
      T product = v[0];
      for (size_t i = 1; i < v.size(); ++i)
        product *= v[i];
      return product;
    }
    
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
