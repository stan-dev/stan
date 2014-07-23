#ifndef STAN__AGRAD__FWD__FUNCTIONS__PHI_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__PHI_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/functions/Phi.hpp>
#include <stan/math/constants.hpp>

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
