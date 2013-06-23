#ifndef __STAN__MATH__REP_ROW_VECTOR_HPP__
#define __STAN__MATH__REP_ROW_VECTOR_HPP__

#include <boost/math/tools/promotion.hpp>
#include <stan/math/validate_non_negative_rep.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                         1,Eigen::Dynamic>
    rep_row_vector(const T& x, int m) {
      validate_non_negative_rep(m,"rep_row_vector");
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           1,Eigen::Dynamic>::Constant(m,x);
    }

  }
}

#endif
