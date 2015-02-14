#ifndef STAN__MATH__FWD__SCAL__FUN__MODIFIED_BESSEL_FIRST_KIND_HPP
#define STAN__MATH__FWD__SCAL__FUN__MODIFIED_BESSEL_FIRST_KIND_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/modified_bessel_first_kind.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    modified_bessel_first_kind(int v, const fvar<T>& z) {
      using stan::math::modified_bessel_first_kind;

      T modified_bessel_first_kind_z(modified_bessel_first_kind(v, z.val_));
      return fvar<T>(modified_bessel_first_kind_z,
                     -v * z.d_ * modified_bessel_first_kind_z / z.val_ 
                     + z.d_ * modified_bessel_first_kind(v - 1,z.val_));
    }
  }
}
#endif
