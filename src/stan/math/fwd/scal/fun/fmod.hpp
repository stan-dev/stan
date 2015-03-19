#ifndef STAN__MATH__FWD__SCAL__FUN__FMOD_HPP
#define STAN__MATH__FWD__SCAL__FUN__FMOD_HPP

#include <stan/math/fwd/core.hpp>

#include <stan/math/prim/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    fmod(const fvar<T>& x1, const fvar<T>& x2) {
      using std::fmod;
      using std::floor;
      return fvar<T>(fmod(x1.val_, x2.val_),
                     x1.d_ - x2.d_ * floor(x1.val_ / x2.val_));
    }

    template <typename T>
    inline
    fvar<T>
    fmod(const fvar<T>& x1, const double x2) {
      using std::fmod;
      using stan::math::value_of;
      if (unlikely(boost::math::isnan(value_of(x1.val_))
                   || boost::math::isnan(x2)))
        return fvar<T>(fmod(x1.val_,x2),stan::math::NOT_A_NUMBER);
      else
        return fvar<T>(fmod(x1.val_, x2), x1.d_ / x2);
    }

    template <typename T>
    inline
    fvar<T>
    fmod(const double x1, const fvar<T>& x2) {
      using std::fmod;
      using std::floor;
      return fvar<T>(fmod(x1, x2.val_), -x2.d_ * floor(x1 / x2.val_));
    }
  }
}
#endif
