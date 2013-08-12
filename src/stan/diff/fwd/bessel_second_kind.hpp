#ifndef __STAN__AGRAD__FWD__BESSEL_SECOND_KIND__HPP__
#define __STAN__AGRAD__FWD__BESSEL_SECOND_KIND__HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/bessel_second_kind.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    bessel_second_kind(int v, const fvar<T>& z) {
      using stan::math::bessel_second_kind;

      T bessel_second_kind_z(bessel_second_kind(v, z.val_));
      return fvar<T>(bessel_second_kind_z,
                     v * z.d_ * bessel_second_kind_z / z.val_ 
                     - z.d_ * bessel_second_kind(v + 1,z.val_));
    }
  }
}
#endif
