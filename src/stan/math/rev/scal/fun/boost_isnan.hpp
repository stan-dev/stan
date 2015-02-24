#ifndef STAN__MATH__REV__SCAL__FUN__BOOST_ISNAN_HPP
#define STAN__MATH__REV__SCAL__FUN__BOOST_ISNAN_HPP

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
    bool isnan(const stan::agrad::var& v) {
      return (boost::math::isnan)(v.val());
    }

  }
}
#endif

