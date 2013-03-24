#ifndef __STAN__MATH__MATRIX__EXP_HPP__
#define __STAN__MATH__MATRIX__EXP_HPP__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Return the element-wise exponentiation of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i,j) = exp(m(i,j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T,Rows,Cols> exp(const Eigen::Matrix<T,Rows,Cols>& m) {
      return m.array().exp().matrix();
    }
    
  }
}
#endif
