#ifndef STAN__MATH__FUNCTIONS__MULTIPLY_LOG_HPP
#define STAN__MATH__FUNCTIONS__MULTIPLY_LOG_HPP

#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /** 
     * Calculated the value of the first argument
     * times log of the second argument while behaving
     * properly with 0 inputs.
     * 
     * \f$ a * \log b \f$.
     * 
     * @param a the first variable
     * @param b the second variable
     * 
     * @return a * log(b)
     */
    template <typename T_a, typename T_b>
    inline typename boost::math::tools::promote_args<T_a,T_b>::type
    multiply_log(const T_a a, const T_b b) {
      using std::log;
      if (b == 0.0 && a == 0.0)
        return 0.0;
      return a * log(b);
    }

  }
}

#endif
