#ifndef __STAN__MATH__FUNCTIONS__GAMMA_P_HPP__
#define __STAN__MATH__FUNCTIONS__GAMMA_P_HPP__

#include <boost/math/special_functions/gamma.hpp>

namespace stan {

  namespace math {

    // throws domain_error if x is at pole
    double gamma_p(double x, double a) {
      return boost::math::gamma_p(x,a);
    }

  }
}

#endif
