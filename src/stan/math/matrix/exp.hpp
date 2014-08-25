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
      T * mat_ = mat.data();
      for (int i = 0, size_ = mat.size(); i < size_; i++)
        mat_[i] = std::exp(mat_[i]);
      return mat;
    }
    
  }
}
#endif
