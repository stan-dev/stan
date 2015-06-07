#ifndef STAN_MATH_PRIM_SCAL_FUN_PRIMITIVE_VALUE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_PRIMITIVE_VALUE_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_arithmetic.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

namespace stan {

  namespace math {

    /**
     * Return the value of the specified arithmetic argument
     * unmodified with its own declared type.
     *
     * <p>This template function can only be instantiated with
     * arithmetic types as defined by Boost's
     * <code>is_arithmetic</code> trait metaprogram.
     *
     * <p>This function differs from <code>stan::math::value_of</code> in that it
     * does not cast all return types to <code>double</code>.
     *
     * @tparam T type of arithmetic input.
     * @param x input.
     * @return input unmodified.
     */
    template <typename T>
    inline
    typename boost::enable_if<boost::is_arithmetic<T>, T>::type
    primitive_value(T x) {
      return x;
    }

    /**
     * Return the primitive value of the specified argument.
     *
     * <p>This implementation only applies to non-arithmetic types as
     * defined by Boost's <code>is_arithmetic</code> trait metaprogram.
     *
     * @tparam T type of non-arithmetic input.
     * @param x input.
     * @return value of input.
     */
    template <typename T>
    inline
    typename boost::disable_if<boost::is_arithmetic<T>, double>::type
    primitive_value(const T& x) {
      using stan::math::value_of;
      return value_of(x);
    }

  }

}

#endif
