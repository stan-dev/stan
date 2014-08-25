#ifndef STAN__MATH__MATRIX__EXP_HPP
#define STAN__MATH__MATRIX__EXP_HPP

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
    inline Eigen::Matrix<T,Rows,Cols> exp(Eigen::Matrix<T,Rows,Cols> mat) {
      for (T * it = mat.data(), * end_ = it + mat.size(); it != end_; it++)
        *it = std::exp(*it);
      return mat;
    }
    
  }
}
#endif
