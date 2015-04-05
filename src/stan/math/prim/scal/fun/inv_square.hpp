#ifndef STAN_MATH_PRIM_SCAL_FUN_INV_SQUARE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_INV_SQUARE_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    template <typename T>
    inline
    typename boost::math::tools::promote_args<T>::type
    inv_square(const T x) {
      return 1.0 / (x * x);
    }
  }
}

#endif
