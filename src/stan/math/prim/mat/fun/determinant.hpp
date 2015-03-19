#ifndef STAN__MATH__PRIM__MAT__FUN__DETERMINANT_HPP
#define STAN__MATH__PRIM__MAT__FUN__DETERMINANT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the determinant of the specified square matrix.
     *
     * @param m Specified matrix.
     * @return Determinant of the matrix.
     * @throw std::domain_error if matrix is not square.
     */
    template <typename T,int R, int C>
    inline T determinant(const Eigen::Matrix<T,R,C>& m) {
      stan::math::check_square("determinant", "m", m);
      return m.determinant();
    }

  }
}
#endif
