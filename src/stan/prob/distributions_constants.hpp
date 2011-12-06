#ifndef __STAN__PROB__DISTRIBUTIONS_CONSTANTS_HPP__
#define __STAN__PROB__DISTRIBUTIONS_CONSTANTS_HPP__

#include <boost/math/constants/constants.hpp>

namespace stan {
  namespace prob {
    using namespace std;
    using namespace stan::maths;
    
    namespace {
   
      const double PI = boost::math::constants::pi<double>();

      const double LOG_ZERO = log(0.0);

      const double LOG_TWO = log(2.0);

      const double NEG_LOG_TWO = -LOG_TWO;

      const double NEG_LOG_SQRT_TWO_PI = - log(sqrt(2.0 * PI));

      const double NEG_LOG_PI = -log(PI);

      const double NEG_LOG_SQRT_PI = -log(sqrt(PI));

      const double NEG_LOG_TWO_OVER_TWO = -LOG_TWO / 2.0;

      const double SQRT_2 = sqrt(2.0);
    }
 
 }
}

#endif
