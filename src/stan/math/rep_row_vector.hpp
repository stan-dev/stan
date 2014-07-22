#ifndef STAN__MATH__REP_ROW_VECTOR_HPP
#define STAN__MATH__REP_ROW_VECTOR_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/error_handling/check_nonnegative.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                         1,Eigen::Dynamic>
    rep_row_vector(const T& x, int m) {
      check_nonnegative("rep_row_vector(%1%)", m,"m", (double*)0);
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           1,Eigen::Dynamic>::Constant(m,x);
    }

  }
}

#endif
