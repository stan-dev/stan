#ifndef STAN__MATH__PRIM__MAT__FUN__REP_VECTOR_HPP
#define STAN__MATH__PRIM__MAT__FUN__REP_VECTOR_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  namespace math {

    template <typename T>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                  Eigen::Dynamic,1>
    rep_vector(const T& x, int n) {
      using stan::math::check_nonnegative;
      check_nonnegative("rep_vector", "n", n);
      return Eigen::Matrix<typename boost::math::tools::promote_args<T>::type,
                           Eigen::Dynamic,1>::Constant(n,x);
    }


  }
}

#endif
