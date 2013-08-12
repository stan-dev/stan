#ifndef __STAN__DIFF__FWD__PHI__HPP__
#define __STAN__DIFF__FWD__PHI__HPP__

#include <stan/diff/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/math/constants.hpp>

namespace stan{

  namespace diff{

    template <typename T>
    inline fvar<T> Phi(const fvar<T>& x) {
      using stan::math::Phi;
      using std::exp;
      using std::sqrt;
      T xv = x.val_;
      return fvar<T>(Phi(xv),
                     exp(xv * xv / -2.0) / sqrt(2.0 * stan::math::pi()));
    }
  }
}
#endif
