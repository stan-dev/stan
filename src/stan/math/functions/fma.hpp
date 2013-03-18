#ifndef __STAN__MATH__FUNCTIONS__FMA_HPP__
#define __STAN__MATH__FUNCTIONS__FMA_HPP__

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * The fused multiply-add operation (C99).  
     *
     * The function is defined by
     *
     * <code>fma(a,b,c) = (a * b) + c</code>.
     *
     * @param a First value.
     * @param b Second value.
     * @param c Third value.
     * @return Product of the first two values plust the third.
     */
    template <typename T1, typename T2, typename T3>
    inline typename boost::math::tools::promote_args<T1,T2,T3>::type
    fma(T1 a, T2 b, T3 c) {
      return (a * b) + c;
    }

  }
}
#endif
