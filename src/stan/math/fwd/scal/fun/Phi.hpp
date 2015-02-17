#ifndef STAN__MATH__FWD__SCAL__FUN__PHI_HPP
#define STAN__MATH__FWD__SCAL__FUN__PHI_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
#include <stan/math/prim/scal/fun/Phi.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline fvar<T> Phi(const fvar<T>& x) {
      using stan::math::Phi;
      using std::exp;
      using std::sqrt;
      T xv = x.val_;
      return fvar<T>(Phi(xv),
                     x.d_ * exp(xv * xv / -2.0) / sqrt(2.0 * stan::math::pi()));
    }
  }
}
#endif
