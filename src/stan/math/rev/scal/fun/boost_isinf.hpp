#ifndef STAN_MATH_REV_SCAL_FUN_BOOST_ISINF_HPP
#define STAN_MATH_REV_SCAL_FUN_BOOST_ISINF_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/rev/core.hpp>

namespace boost {

  namespace math {

    /**
     * Checks if the given number is infinite.
     *
     * Return <code>true</code> if the specified variable's
     * value is infinite.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is infinite.
     */
    template <>
    inline
    bool isinf(const stan::math::var& v) {
      return (boost::math::isinf)(v.val());
    }

  }
}
#endif

