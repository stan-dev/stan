#ifndef STAN__MATH__REP_VECTOR_HPP
#define STAN__MATH__REP_VECTOR_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                  Eigen::Dynamic,1>
    rep_vector(const T& x, int n) {
      using stan::error_handling::check_nonnegative;
      check_nonnegative("rep_vector", "n", n);
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           Eigen::Dynamic,1>::Constant(n,x);
    }


  }
}

#endif
