#ifndef __STAN__AGRAD__FWD__FUNCTIONS__EXP_HPP__
#define __STAN__AGRAD__FWD__FUNCTIONS__EXP_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {


    template <typename T>
    inline
    fvar<T>
    exp(const fvar<T>& x) {
      using std::exp;
      return fvar<T>(exp(x.val_), x.d_ * exp(x.val_));
    }
  }
}
#endif
