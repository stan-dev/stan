#ifndef STAN__MATH__PRIM__MAT__FUN__DIAGONAL_HPP
#define STAN__MATH__PRIM__MAT__FUN__DIAGONAL_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace math {

    /**
     * Return a column vector of the diagonal elements of the
     * specified matrix.  The matrix is not required to be square.
     * @param m Specified matrix.
     * @return Diagonal of the matrix.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    diagonal(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      return m.diagonal();
    }

  }
}
#endif
