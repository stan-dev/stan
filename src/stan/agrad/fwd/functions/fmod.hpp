#ifndef STAN__AGRAD__FWD__FUNCTIONS__FMOD_HPP
#define STAN__AGRAD__FWD__FUNCTIONS__FMOD_HPP

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

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
