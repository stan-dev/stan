#ifndef STAN__MATH__FUNCTIONS__INV_SQRT_HPP
#define STAN__MATH__FUNCTIONS__INV_SQRT_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {
    
    template <typename T>
    inline 
    typename boost::math::tools::promote_args<T>::type
    inv_sqrt(const T x) {
      using std::sqrt;

      return 1.0 / sqrt(x);
    }

  }
}

#endif
