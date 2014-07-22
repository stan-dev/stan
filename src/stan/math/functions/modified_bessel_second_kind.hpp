#ifndef STAN__MATH__FUNCTIONS__MODIFIED_BESSEL_SECOND_KIND_HPP
#define STAN__MATH__FUNCTIONS__MODIFIED_BESSEL_SECOND_KIND_HPP

#include <boost/math/special_functions/bessel.hpp>

namespace stan {
  namespace math {

    template<typename T2>
    inline T2 
    modified_bessel_second_kind(const int v, const T2 z) { 
      return boost::math::cyl_bessel_k(v,z); 
    }

  }
}

#endif
