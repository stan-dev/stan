#ifndef STAN__MATH__FUNCTIONS__LOG2_HPP
#define STAN__MATH__FUNCTIONS__LOG2_HPP

#include <stdexcept>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/constants.hpp>

namespace stan {

  namespace math {

    /**
     * Returns the base 2 logarithm of the argument (C99).
     *
     * The function is defined by:
     *
     * <code>log2(a) = log(a) / std::log(2.0)</code>.
     *
     * @tparam T type of scalar
     * @param a Value.
     * @return Base 2 logarithm of the value.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log2(const T a) {
      using std::log;
      return log(a) / LOG_2;
    }

    /**
     * Return natural logarithm of two.
     *
     * @return Natural logarithm of two.
     */
    inline double log2() {
      return LOG_2;
    }

  }
}

#endif
