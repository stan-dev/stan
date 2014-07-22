#ifndef STAN__MATH__FUNCTIONS__LOG1M_HPP
#define STAN__MATH__FUNCTIONS__LOG1M_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/functions/log1p.hpp>

namespace stan {
  namespace math {

    /**
     * Return the natural logarithm of one minus the specified value.
     *
     * The main use of this function is to cut down on intermediate
     * values during algorithmic differentiation.
     *
     * @param x Specified value.
     * @return Natural log of one minus <code>x</code>.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1m(T x) {
      return log1p(-x);
    }

  }
}

#endif
