#ifndef __STAN__MATH__MATRIX__TO_VECTOR_HPP__
#define __STAN__MATH__MATRIX__TO_VECTOR_HPP__

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T, int R1, int R2>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    to_vector(const Eigen::Matrix<T,R1,R2>& m) {
      return Eigen::Matrix<T,Eigen::Dynamic,1>::Map(m.data(), m.rows()*m.cols());
    }

  }
}
#endif
