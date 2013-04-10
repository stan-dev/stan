#ifndef __STAN__MATH__REP_ROW_VECTOR_HPP__
#define __STAN__MATH__REP_ROW_VECTOR_HPP__

#include <stdexcept>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>

#include <stan/math/rep_array.hpp>

namespace stan {
  namespace math {

    template <typename T>
    inline Eigen::Matrix<T,1,Eigen::Dynamic>
    rep_row_vector(const T& x, int m) {
      validate_non_negative_rep(m,"rep_row_vector");
      return Eigen::Matrix<T,1,Eigen::Dynamic>::Constant(m,x);
    }

  }
}

#endif
