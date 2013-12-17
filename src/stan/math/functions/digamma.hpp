#ifndef __STAN__MATH__FUNCTIONS__DIGAMMA_HPP__
#define __STAN__MATH__FUNCTIONS__DIGAMMA_HPP__

#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    double digamma(double x) {
      return boost::math::digamma(x);
    }

  }
}

#endif
