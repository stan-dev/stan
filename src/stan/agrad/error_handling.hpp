#ifndef __STAN__AGRAD__ERROR_HANDLING_HPP__
#define __STAN__AGRAD__ERROR_HANDLING_HPP__

// FIXME: fill in agrad-specific error handling

#include <stan/maths/error_handling.hpp>

namespace boost {

  namespace math {

    /**
     * Return <code>true</code> the floating point type
     * for the specified variable's value.
     *
     * @param v Variable to classify.
     * @return Classification of value of the variable.
     */
    template <>
    inline
    int fpclassify(const stan::agrad::var v) {
      return boost::math::fpclassify(v.val());
    }

    /**
     * Return <code>true</code> if the specified variable's
     * value is finite.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is finite.
     */
    template <>
    inline
    bool isfinite(const stan::agrad::var v) {
      return boost::math::isfinite(v.val());
    }

    /**
     * Return <code>true</code> if the specified variable's
     * value is infinite.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is infinite.
     */
    template <>
    inline
    bool isinf(const stan::agrad::var v) {
      return boost::math::isinf(v.val());
    }

    /**
     * Return <code>true</code> if the specified variable
     * has a value that is NaN.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is NaN.
     */
    template <>
    inline
    bool isnan(const stan::agrad::var v) {
      return boost::math::isnan(v.val());
    }

    /**
     * Return <code>true</code> if the specified variable
     * has a value that is NaN.
     *
     * @param v Variable to test.
     * @return <code>true</code> if variable is NaN.
     */
    template <>
    inline
    bool isnormal(const stan::agrad::var v) {
      return boost::math::isnormal(v.val());
    }

  }
}
  

#endif

