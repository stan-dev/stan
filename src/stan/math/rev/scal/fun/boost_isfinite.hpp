#ifndef STAN__MATH__REV__SCAL__FUN__BOOST_ISFINITE_HPP
#define STAN__MATH__REV__SCAL__FUN__BOOST_ISFINITE_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/rev/core/var.hpp>

namespace boost {

  namespace math {

    /**
     * Checks if the given number has finite value.
     *
     * Return <code>true</code> if the specified variable's
     * value is finite.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is finite.
     */
    template <>
    inline
    bool isfinite(const stan::agrad::var& v) {
      return (boost::math::isfinite)(v.val());
    }

  }
}
#endif

