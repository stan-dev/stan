#ifndef STAN__MATH__MATRIX__DIAG_PRE_MULTIPLY_HPP
#define STAN__MATH__MATRIX__DIAG_PRE_MULTIPLY_HPP

#include <stdexcept>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1,int C1,int R2,int C2>
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R2, C2>
    diag_pre_multiply(const Eigen::Matrix<T1,R1,C1>& m1,
                  const Eigen::Matrix<T2,R2,C2>& m2) {
      if (m1.cols() != 1 && m1.rows() != 1)
        throw std::domain_error("m1 must be a vector");
      int m2_rows = m2.rows();
      if (m1.size() != m2_rows)
        throw std::domain_error("m1 must have same length as m2 has rows");
      int m2_cols = m2.cols();
      Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type, R2, C2>
        result(m2_rows,m2_cols);
      for (int j = 0; j < m2_cols; ++j)
        for (int i = 0; i < m2_rows; ++i)
          result(i,j) = m1(i) * m2(i,j);
      return result;
    }

  }
}
#endif
