#ifndef __STAN__AGRAD__FWD__FUNCTIONS__CEIL_HPP__
#define __STAN__AGRAD__FWD__FUNCTIONS__CEIL_HPP__

#include <stan/agrad/fwd/fvar.hpp>
#include <stan/meta/traits.hpp>

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    fvar<T>
    ceil(const fvar<T>& x) {
      using std::ceil;
        return fvar<T>(ceil(x.val_), 0);
    }
  }
}
#endif
