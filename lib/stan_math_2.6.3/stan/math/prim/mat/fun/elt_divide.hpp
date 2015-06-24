#ifndef STAN_MATH_PRIM_MAT_FUN_ELT_DIVIDE_HPP
#define STAN_MATH_PRIM_MAT_FUN_ELT_DIVIDE_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>

namespace stan {
  namespace math {

    /**
     * Return the elementwise division of the specified matrices.
     *
     * @tparam T1 Type of scalars in first matrix.
     * @tparam T2 Type of scalars in second matrix.
     * @tparam R Row type of both matrices.
     * @tparam C Column type of both matrices.
     * @param m1 First matrix
     * @param m2 Second matrix
     * @return Elementwise division of matrices.
     */
    template <typename T1, typename T2, int R, int C>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type, R, C>
    elt_divide(const Eigen::Matrix<T1, R, C>& m1,
               const Eigen::Matrix<T2, R, C>& m2) {
      stan::math::check_matching_dims("elt_divide",
                                                "m1", m1,
                                                "m2", m2);
      Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
                    R, C> result(m1.rows(), m2.cols());
      for (int i = 0; i < m1.size(); ++i)
        result(i) = m1(i) / m2(i);
      return result;
    }

    /**
     * Return the elementwise division of the specified matrix
     * by the specified scalar.
     *
     * @tparam T1 Type of scalars in the matrix.
     * @tparam T2 Type of the scalar.
     * @tparam R Row type of the matrix.
     * @tparam C Column type of the matrix.
     * @param m matrix
     * @param s scalar
     * @return Elementwise division of a scalar by matrix.
     */
    template <typename T1, typename T2, int R, int C>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type, R, C>
    elt_divide(const Eigen::Matrix<T1, R, C>& m, T2 s) {
      return m / s;
    }

     /**
     * Return the elementwise division of the specified scalar
     * by the specified matrix.
     *
     * @tparam T1 Type of the scalar.
     * @tparam T2 Type of scalars in the matrix.
     * @tparam R Row type of the matrix.
     * @tparam C Column type of the matrix.
     * @param s scalar
     * @param m matrix
     * @return Elementwise division of a scalar by matrix.
     */
    template <typename T1, typename T2, int R, int C>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type, R, C>
    elt_divide(T1 s,
               const Eigen::Matrix<T2, R, C>& m) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
                    R, C> result(m.rows(), m.cols());
      for (int i = 0; i < m.size(); ++i)
        result(i) = s / m(i);
      return result;
    }

  }
}
#endif
