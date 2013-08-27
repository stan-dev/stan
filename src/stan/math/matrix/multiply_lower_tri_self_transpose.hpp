#ifndef __STAN__MATH__MATRIX__MULTIPLY_LOWER_TRI_SELF_HPP__
#define __STAN__MATH__MATRIX__MULTIPLY_LOWER_TRI_SELF_HPP__

#include <stan/math/matrix/typedefs.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the result of multiplying the lower triangular
     * portion of the input matrix by its own transpose.
     * @param L Matrix to multiply.
     * @return The lower triangular values in L times their own
     * transpose.
     * @throw std::domain_error If the input matrix is not square.
     */
    inline matrix_d
    multiply_lower_tri_self_transpose(const matrix_d& L) {
      if (L.rows() == 0)
        return matrix_d(0,0);
      if (L.rows() == 1) {
        matrix_d result(1,1);
        result(0,0) = L(0,0) * L(0,0);
        return result;
      }
      // FIXME:  write custom following diff/matrix because can't get L_tri into
      // multiplication as no template support for tri * tri
      matrix_d L_tri = L.transpose().triangularView<Eigen::Upper>();
      return L.triangularView<Eigen::Lower>() * L_tri;
    }

  }
}
#endif
