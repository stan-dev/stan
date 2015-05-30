#ifndef STAN_MATH_REV_SCAL_FUN_BOOST_ISNAN_HPP
#define STAN_MATH_REV_SCAL_FUN_BOOST_ISNAN_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/rev/core.hpp>

namespace boost {

  namespace math {

    /**
     * Checks if the given number is NaN
     *
     * Return <code>true</code> if the specified variable
     * has a value that is NaN.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is NaN.
     */
    template <>
    inline
    bool isnan(const stan::math::var& v) {
      return (boost::math::isnan)(v.val());
    }

  }
}
#endif

