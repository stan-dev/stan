#ifndef __STAN__MATH__REP_VECTOR_HPP__
#define __STAN__MATH__REP_VECTOR_HPP__

#include <boost/math/tools/promotion.hpp>
#include <stan/math/validate_non_negative_rep.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                  Eigen::Dynamic,1>
    rep_vector(const T& x, int n) {
      validate_non_negative_rep(n,"rep_vector");
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           Eigen::Dynamic,1>::Constant(n,x);
    }


  }
}

#endif
