#ifndef STAN__MATH__FUNCTIONS__BESSEL_FIRST_KIND_HPP
#define STAN__MATH__FUNCTIONS__BESSEL_FIRST_KIND_HPP

#include <boost/math/special_functions/bessel.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>

namespace stan {
  namespace math {

    template<typename T2>
    inline T2 
    bessel_first_kind(const int v, const T2 z) { 
      check_not_nan("bessel_first_kind(%1%)", z, "z", static_cast<double*>(0));
      return boost::math::cyl_bessel_j(v,z); 
    }

  }
}

#endif
