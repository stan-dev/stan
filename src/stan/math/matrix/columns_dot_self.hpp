#ifndef __STAN__MATH__MATRIX__COLUMNS_DOT_SELF_HPP__
#define __STAN__MATH__MATRIX__COLUMNS_DOT_SELF_HPP__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Returns the dot product of each column of a matrix with itself.
     * @param x Matrix.
     * @tparam T scalar type
     */
    template<typename T,int R,int C>
    inline Eigen::Matrix<T,1,C> 
    columns_dot_self(const Eigen::Matrix<T,R,C>& x) {
      return x.colwise().squaredNorm();
    }
    
  }
}
#endif
