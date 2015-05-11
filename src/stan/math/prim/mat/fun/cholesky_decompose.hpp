#ifndef STAN_MATH_PRIM_MAT_FUN_CHOLESKY_DECOMPOSE_HPP
#define STAN_MATH_PRIM_MAT_FUN_CHOLESKY_DECOMPOSE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>

namespace stan {
  namespace math {

    /**
     * Return the lower-triangular Cholesky factor (i.e., matrix
     * square root) of the specified square, symmetric matrix.  The return
     * value \f$L\f$ will be a lower-traingular matrix such that the
     * original matrix \f$A\f$ is given by
     * <p>\f$A = L \times L^T\f$.
     * @param m Symmetrix matrix.
     * @return Square root of matrix.
     * @throw std::domain_error if m is not a symmetric matrix or
     *   if m is not positive definite (if m has more than 0 elements)
     */
    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
    cholesky_decompose(const Eigen::Matrix
                       <T, Eigen::Dynamic, Eigen::Dynamic>& m) {
      stan::math::check_square("cholesky_decompose", "m", m);
      stan::math::check_symmetric("cholesky_decompose", "m", m);
      Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> >
        llt(m.rows());
      llt.compute(m);

      stan::math::check_pos_definite("cholesky_decompose", "m", llt);
      
      return llt.matrixL();
    }

  }
}
#endif
