#ifndef __STAN__MATH__FUNCTIONS__BESSEL_FIRST_KIND_HPP__
#define __STAN__MATH__FUNCTIONS__BESSEL_FIRST_KIND_HPP__

#include <boost/math/special_functions/bessel.hpp>

namespace stan {
  namespace math {

    template<typename T2>
    inline T2 
    bessel_first_kind(const int v, const T2 z) { 
      return boost::math::cyl_bessel_j(v,z); 
    }

  }
}

#endif
