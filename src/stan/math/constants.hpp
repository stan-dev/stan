#ifndef __STAN__MATH__CONSTANTSS_HPP__
#define __STAN__MATH__CONSTANTSS_HPP__

#include <boost/math/constants/constants.hpp>

namespace stan {

  namespace math {

    /**
     * The base of the natural logarithm, 
     * \f$ e \f$.
     */
    const double E = boost::math::constants::e<double>();

    /** 
     * The value of pi, 
     * $\f \pi \f$.
     */
    const double PI = boost::math::constants::pi<double>();

    /** 
     * The value of the square root of 2, 
     * \$f \sqrt{2} \f$. 
     */
    const double SQRT_2 = std::sqrt(2.0);

    /**
     * The natural logarithm of 2, 
     * \f$ \log 2 \f$.
     */
    const double LOG_2 = std::log(2.0);

    /**
     * The natural logarithm of 10, 
     * $\f \log 10 \f$..
     */
    const double LOG_10 = std::log(10.0);

    /**
     * Positive infinity.
     */
    const double INFTY = std::numeric_limits<double>::infinity();

    /**
     * Negative infinity.
     */
    const double NEGATIVE_INFTY = - std::numeric_limits<double>::infinity();

    /**
     * (Quiet) not-a-number value.
     */
    const double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();
    
    /**
     * Smallest positive value.
     */
    const double EPSILON = std::numeric_limits<double>::epsilon();

    /**
     * Largest negative value (i.e., smallest absolute value).
     */
    const double NEGATIVE_EPSILON = - std::numeric_limits<double>::epsilon();

  }
}

#endif
