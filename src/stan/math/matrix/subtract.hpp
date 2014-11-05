#ifndef STAN__MATH__MATRIX__SUBTRACT_HPP
#define STAN__MATH__MATRIX__SUBTRACT_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_matching_dims.hpp>

namespace stan {
  namespace math {

    /**
     * Return the result of subtracting the second specified matrix
     * from the first specified matrix.  The return scalar type is the
     * promotion of the input types.
     *
     * @tparam T1 Scalar type of first matrix.
     * @tparam T2 Scalar type of second matrix.
     * @tparam R Row type of matrices.
     * @tparam C Column type of matrices.
     * @param m1 First matrix.
     * @param m2 Second matrix.
     * @return Difference between first matrix and second matrix.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1,R,C>& m1,
             const Eigen::Matrix<T2,R,C>& m2) {
      stan::error_handling::check_matching_dims("subtract",
                                                "m1", m1,
                                                "m2", m2);
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m1.rows(), m1.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m1(i) - m2(i);
      return result;
    }

    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const T1& c,
             const Eigen::Matrix<T2,R,C>& m) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        result(i) = c - m(i);
      return result;
    }

    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
    subtract(const Eigen::Matrix<T1,R,C>& m,
             const T2& c) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R, C>
        result(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        result(i) = m(i) - c;
      return result;
    }

  }
}
#endif
