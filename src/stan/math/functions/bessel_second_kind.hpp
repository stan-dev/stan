#ifndef __STAN__MATH__FUNCTIONS__BESSEL_SECOND_KIND_HPP__
#define __STAN__MATH__FUNCTIONS__BESSEL_SECOND_KIND_HPP__

#include <boost/math/special_functions/bessel.hpp>

namespace stan {
  namespace math {

    template <typename T2>
    inline T2
    bessel_second_kind(const int v, const T2 z) { 
      return boost::math::cyl_neumann(v,z); 
    }

  }
}

#endif
