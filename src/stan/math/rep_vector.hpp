#ifndef __STAN__MATH__REP_VECTOR_HPP__
#define __STAN__MATH__REP_VECTOR_HPP__

#include <stdexcept>
#include <vector>

#include <stan/math/matrix/Eigen.hpp>

#include <stan/math/rep_array.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline Eigen::Matrix<T,Eigen::Dynamic,1>
    rep_vector(const T& x, int n) {
      validate_non_negative_rep(n,"rep_vector");
      return Eigen::Matrix<T,Eigen::Dynamic,1>::Constant(n,x);
    }

  }
}

#endif
