#ifndef STAN__MATH__PRIM__MAT__FUN__READ_CORR_MATRIX_HPP
#define STAN__MATH__PRIM__MAT__FUN__READ_CORR_MATRIX_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/read_corr_L.hpp>
#include <stan/math/prim/mat/fun/multiply_lower_tri_self_transpose.hpp>

namespace stan {

  namespace prob {


    /**
     * Return the correlation matrix of the specified dimensionality
     * corresponding to the specified canonical partial correlations.
     *
     * <p>See <code>read_corr_matrix(Array,size_t,T)</code>
     * for more information.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @return Cholesky factor of correlation matrix for specified
     * canonical partial correlations.
     * @tparam T Type of underlying scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                     const size_t K) {
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L
        = read_corr_L(CPCs, K);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }

    /**
     * Return the correlation matrix of the specified dimensionality
     * corresponding to the specified canonical partial correlations,
     * incrementing the specified scalar reference with the log
     * absolute determinant of the Jacobian of the transformation.
     *
     * It is usually preferable to utilize the version that returns
     * the Cholesky factor of the correlation matrix rather than the
     * correlation matrix itself in statistical calculations.
     *
     * @param CPCs The (K choose 2) canonical partial correlations in
     * (-1,1).
     * @param K Dimensionality of correlation matrix.
     * @param log_prob Reference to variable to increment with the log
     * Jacobian determinant.
     * @return Correlation matrix for specified partial correlations.
     * @tparam T Type of underlying scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    read_corr_matrix(const Eigen::Array<T,Eigen::Dynamic,1>& CPCs,
                     const size_t K,
                     T& log_prob) {

      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> L
        = read_corr_L(CPCs, K, log_prob);
      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(L);
    }

  }

}

#endif
