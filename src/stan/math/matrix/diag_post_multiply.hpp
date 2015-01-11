#ifndef STAN__MATH__MATRIX__DIAG_POST_MULTIPLY_HPP
#define STAN__MATH__MATRIX__DIAG_POST_MULTIPLY_HPP

#include <stdexcept>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R1, C1>
    diag_post_multiply(const Eigen::Matrix<T1,R1,C1>& m1,
                  const Eigen::Matrix<T2,R2,C2>& m2) {
      if (m2.cols() != 1 && m2.rows() != 1)
        throw std::domain_error("m2 must be a vector");
      int m1_cols = m1.cols();
      if (m2.size() != m1_cols)
        throw std::domain_error("m2 must have same length as m1 has columns");
      int m1_rows = m1.rows();
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R1, C1>
        result(m1_rows, m1_cols);
      
      for (int j = 0; j < m1_cols; ++j)
        for (int i = 0; i < m1_rows; ++i)
          result(i,j) = m2(j) * m1(i,j);
      return result;
    }

  }
}
#endif
