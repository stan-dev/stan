#ifndef __STAN__MATH__SPECIAL_FUNCTIONS_HPP__
#define __STAN__MATH__SPECIAL_FUNCTIONS_HPP__

#include <stdexcept>

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/tools/promotion.hpp>
#include <boost/throw_exception.hpp>

#include <stan/math/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/inv_logit.hpp>

namespace stan {

  namespace math {

    /** 
     * Return the scalar value and ignore the remaining
     * arguments.
     *
     * <p>This function provides an overload of
     * <code>simple_var</code> to use with primitive values for which
     * the type and derivative type are the same.  The other overloads
     * are for <code>stan::agrad::var</code> arguments; the
     * definitions can be found in
     * <code>stan/agrad/partials_vari.hpp</code>.
     * 
     * @tparam T1 Type of first dummy argument and derivative.
     * @tparam T2 Type of second dummy argument and derivative.
     * @tparam T3 Type of third dummy argument and derivative.
     * @param v Value to return.
     * @return Value.
     */
    template <typename T1, typename T2, typename T3>
    inline double simple_var(double v, 
                             const T1& /*y1*/, const T1& /*dy1*/, 
                             const T2& /*y2*/, const T2& /*dy2*/,
                             const T3& /*y3*/, const T3& /*dy3*/) {
      return v;
    }


    // CONSTANTS

    /**
     * Return the value of pi.
     * 
     * @return Pi.
     */
    inline double pi() {
      return boost::math::constants::pi<double>();
    }

    /**
     * Return the base of the natural logarithm.
     *
     * @return Base of natural logarithm.
     */
    inline double e() {
      return E;
    }

    /**
     * Return the square root of two.
     *
     * @return Square root of two. 
     */
    inline double sqrt2() {
      return SQRT_2;
    }


    /**
     * Return natural logarithm of ten.
     *
     * @return Natural logarithm of ten.
     */
    inline double log10() {
      return LOG_10;
    }

    /**
     * Return positive infinity.
     *
     * @return Positive infinity.
     */
    inline double positive_infinity() {
      return INFTY;
    }

    /**
     * Return negative infinity.
     *
     * @return Negative infinity.
     */
    inline double negative_infinity() {
      return NEGATIVE_INFTY;
    }

    /**
     * Return (quiet) not-a-number.
     *
     * @return Quiet not-a-number.
     */
    inline double not_a_number() {
      return NOT_A_NUMBER;
    }

    /**
     * Return minimum positive number representable.
     *
     * @return Minimum positive number.
     */
    inline double epsilon() {
      return EPSILON;
    }

    /**
     * Return maximum negative number (i.e., negative
     * number with smallest absolute value).
     *
     * @return Maximum negative number.
     */
    inline double negative_epsilon() {
      return NEGATIVE_EPSILON;
    }

    /**
     * Return an integer with an equivalent boolean value to specified
     * input.  For integers, this reduces to the identity function.
     *
     * @param x value.
     * @return The value.
     */
    inline int as_bool(int x) {
      return x;
    }
    /**
     * Return 1 if the argument is unequal to zero and 0 otherwise.
     *
     * @param x Value.
     * @return 1 if argument is equal to zero and 0 otherwise.
     */
    inline int as_bool(double x) {
      return x != 0.0;
    }

  }

}

#endif
