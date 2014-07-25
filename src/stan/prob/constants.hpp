#ifndef STAN__PROB__DISTRIBUTIONS_CONSTANTS_HPP
#define STAN__PROB__DISTRIBUTIONS_CONSTANTS_HPP

#include <boost/math/constants/constants.hpp>

namespace stan {

  namespace prob {

    
    namespace {

      const double LOG_PI = std::log(boost::math::constants::pi<double>());

      const double LOG_ZERO = std::log(0.0);

      const double LOG_TWO = std::log(2.0);

      const double NEG_LOG_TWO = - LOG_TWO;

      const double NEG_LOG_SQRT_TWO_PI 
      = - std::log(std::sqrt(2.0 * boost::math::constants::pi<double>()));

      const double NEG_LOG_PI = - LOG_PI;

      const double NEG_LOG_SQRT_PI 
      = -std::log(std::sqrt(boost::math::constants::pi<double>()));
      
      const double NEG_LOG_TWO_OVER_TWO = - LOG_TWO / 2.0;

      const double SQRT_2 = std::sqrt(2.0);

      const double LOG_TWO_PI = LOG_TWO + LOG_PI;
      
      const double NEG_LOG_TWO_PI = - LOG_TWO_PI;
    }
 
 }

}

#endif
