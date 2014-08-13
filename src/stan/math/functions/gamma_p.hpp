#ifndef STAN__MATH__FUNCTIONS__GAMMA_P_HPP
#define STAN__MATH__FUNCTIONS__GAMMA_P_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <cmath>
#include <ostream>

#include "boost/format/alt_sstream.hpp"
#include "boost/format/alt_sstream_impl.hpp"
#include "boost/format/format_implementation.hpp"
#include "boost/optional/optional.hpp"

namespace stan {

  namespace math {

    // throws domain_error if x is at pole
    double gamma_p(double x, double a) {
      return boost::math::gamma_p(x,a);
    }

  }
}

#endif
