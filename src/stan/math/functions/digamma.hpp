#ifndef STAN__MATH__FUNCTIONS__DIGAMMA_HPP
#define STAN__MATH__FUNCTIONS__DIGAMMA_HPP

#include <boost/math/special_functions/digamma.hpp>

namespace stan {

  namespace math {

    double digamma(double x) {
      return boost::math::digamma(x);
    }

  }
}

#endif
