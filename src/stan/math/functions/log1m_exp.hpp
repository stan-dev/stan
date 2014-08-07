#ifndef STAN__MATH__FUNCTIONS__LOG1M_EXP_HPP
#define STAN__MATH__FUNCTIONS__LOG1M_EXP_HPP

#include <boost/math/tools/promotion.hpp>
#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <stan/math/functions/log1m.hpp>

namespace stan {
  namespace math {

    /**
     * Calculates the log of 1 minus the exponential of the specified
     * value without overflow log1m_exp(x) = log(1-exp(x)).
     *
     * This function is only defined for x<0
     *
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    log1m_exp(const T a) {
      if (a >= 0)
        return std::numeric_limits<double>::quiet_NaN();
      else if (a > -0.693147)
        return std::log(-boost::math::expm1(a)); //0.693147 is approximatelly equal to log(2)
      else
        return log1m(std::exp(a));
    }

  }
}

#endif
