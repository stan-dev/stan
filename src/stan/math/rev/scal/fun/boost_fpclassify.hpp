#ifndef STAN__MATH__REV__SCAL__FUN__BOOST_FPCLASSIFY_HPP
#define STAN__MATH__REV__SCAL__FUN__BOOST_FPCLASSIFY_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/rev/core/var.hpp>

namespace boost {

  namespace math {

    /**
     * Categorizes the given stan::agrad::var value.
     * 
     * Categorizes the stan::agrad::var value, v, into the following categories:
     * zero, subnormal, normal, infinite, or NAN.
     *
     * @param v Variable to classify.
     * @return One of <code>FP_ZERO</code>, <code>FP_NORMAL</code>, 
     *   <code>FP_FINITE</code>, <code>FP_INFINITE</code>, <code>FP_NAN</code>,
     *   or <code>FP_SUBZERO</code>, specifying the category of v.
     */
    template <>
    inline
    int fpclassify(const stan::agrad::var& v) {
      return (boost::math::fpclassify)(v.val());
    }

  }
}
#endif

