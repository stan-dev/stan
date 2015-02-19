#ifndef STAN__MATH__FWD__SCAL__FUN__BESSEL_FIRST_KIND_HPP
#define STAN__MATH__FWD__SCAL__FUN__BESSEL_FIRST_KIND_HPP

#include <stan/math/fwd/core/fvar.hpp>

#include <stan/math/prim/scal/fun/bessel_first_kind.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    bessel_first_kind(int v, const fvar<T>& z) {
      using stan::math::bessel_first_kind;

      T bessel_first_kind_z(bessel_first_kind(v, z.val_));
      return fvar<T>(bessel_first_kind_z,
                     v * z.d_ * bessel_first_kind_z / z.val_ 
                     - z.d_ * bessel_first_kind(v + 1,z.val_));
    }
  }
}
#endif
