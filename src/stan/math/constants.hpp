#ifndef __STAN__MATH__CONSTANTS_HPP__
#define __STAN__MATH__CONSTANTS_HPP__

#include <boost/math/constants/constants.hpp>

namespace stan {

  namespace math {

    /**
     * The base of the natural logarithm, 
     * \f$ e \f$.
     */
    const double E = boost::math::constants::e<double>();

    /** 
     * The value of the square root of 2, 
     * \f$ \sqrt{2} \f$. 
     */
    const double SQRT_2 = std::sqrt(2.0);

    /** 
     * The value of 1 over the square root of 2, 
     * \f$ 1 / \sqrt{2} \f$. 
     */
    const double INV_SQRT_2 = 1.0 / SQRT_2;

    /**
     * The natural logarithm of 2, 
     * \f$ \log 2 \f$.
     */
    const double LOG_2 = std::log(2.0);

    /**
     * The natural logarithm of 10, 
     * \f$ \log 10 \f$.
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


    /*
     * Log pi divided by 4
     * \f$ \log \pi / 4 \f$
     */
    const double LOG_PI_OVER_FOUR = std::log(boost::math::constants::pi<double>()) / 4.0;
  }
}

#endif
