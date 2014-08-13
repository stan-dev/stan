#ifndef STAN__MATH__FUNCTIONS__LGAMMA_HPP
#define STAN__MATH__FUNCTIONS__LGAMMA_HPP

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
    double lgamma(double x) {
      return boost::math::lgamma(x);
    }

  }
}

#endif
