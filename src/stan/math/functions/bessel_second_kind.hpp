#ifndef __STAN__MATH__FUNCTIONS__BESSEL_SECOND_KIND_HPP__
#define __STAN__MATH__FUNCTIONS__BESSEL_SECOND_KIND_HPP__

#include <boost/math/special_functions/bessel.hpp>

namespace stan {
  namespace math {

    template<typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type 
    bessel_second_kind(const T1 v, const T2 z) { 
      return boost::math::cyl_neumann(v,z); 
    }

  }
}

#endif
