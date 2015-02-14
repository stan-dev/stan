#ifndef STAN__MATH__PRIM__MAT__FUN__LOG_HPP
#define STAN__MATH__PRIM__MAT__FUN__LOG_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {
    
    /**
     * Return the element-wise logarithm of the matrix or vector.
     *
     * @param m The matrix or vector.
     * @return ret(i,j) = log(m(i,j))
     */
    template<typename T, int Rows, int Cols>
    inline Eigen::Matrix<T,Rows,Cols> log(const Eigen::Matrix<T,Rows,Cols>& m) {
      return m.array().log().matrix();
    }

    
  }
}
#endif
