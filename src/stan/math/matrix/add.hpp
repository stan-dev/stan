#ifndef STAN__MATH__MATRIX__ADD_HPP
#define STAN__MATH__MATRIX__ADD_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_matching_dims.hpp>

namespace stan {
  namespace math {

    /**
     * Return the sum of the specified matrices.  The two matrices
     * must have the same dimensions. 
     * @tparam T1 Scalar type of first matrix.
     * @tparam T2 Scalar type of second matrix.
     * @tparam R Row type of matrices.
     * @tparam C Column type of matrices.
     * @param m1 First matrix.
     * @param m2 Second matrix.  
     * @return Sum of the matrices.
     * @throw std::domain_error if m1 and m2 do not have the same
     * dimensions.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const Eigen::Matrix<T1,R,C>& m1,
        const Eigen::Matrix<T2,R,C>& m2) {
      stan::error_handling::check_matching_dims("add",
                                                "m1", m1,
                                                "m2", m2);
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m1.rows(),m1.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m1(i) + m2(i);
      return result;
    }
    
    /**
     * Return the sum of the specified matrix and specified scalar.
     *
     * @tparam T1 Scalar type of matrix.
     * @tparam T2 Type of scalar.
     * @param m Matrix.
     * @param c Scalar.
     * @return The matrix plus the scalar.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const Eigen::Matrix<T1,R,C>& m, 
        const T2& c) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m.rows(),m.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = m(i) + c;
      return result;
    }
    
    /**
     * Return the sum of the specified scalar and specified matrix.
     *
     * @tparam T1 Type of scalar.
     * @tparam T2 Scalar type of matrix.
     * @param c Scalar.
     * @param m Matrix.
     * @return The scalar plus the matrix.
     */
    template <typename T1, typename T2, int R, int C>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>
    add(const T1& c,
        const Eigen::Matrix<T2,R,C>& m) {
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R,C>      
        result(m.rows(),m.cols());
      for (int i = 0; i < result.size(); ++i)
        result(i) = c + m(i);
      return result;
    }

  }
}
#endif
