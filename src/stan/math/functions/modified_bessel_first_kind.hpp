#ifndef STAN__MATH__FUNCTIONS__MODIFIED_BESSEL_FIRST_KIND_HPP
#define STAN__MATH__FUNCTIONS__MODIFIED_BESSEL_FIRST_KIND_HPP

#include <boost/math/special_functions/bessel.hpp>
#include <stan/math/error_handling/check_not_nan.hpp>

namespace stan {
  namespace math {

    template<typename T2>
    inline T2 
    modified_bessel_first_kind(const int v, const T2 z) { 
      check_not_nan("modified_bessel_first_kind(%1%)", z, "z", static_cast<double*>(0));

      return boost::math::cyl_bessel_i(v,z); 
    }

  }
}

#endif
