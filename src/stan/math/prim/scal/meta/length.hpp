#ifndef STAN__MATH__PRIM__SCAL__META__LENGTH_HPP
#define STAN__MATH__PRIM__SCAL__META__LENGTH_HPP

#include <cstdlib>
#include <cstddef>

namespace stan {

  size_t length(double /*x*/) {
    return 1U;
  }
  size_t length(int /*x*/) {
    return 1U;
  }
  size_t length(size_t /*x*/) {
    return 1U;
  }
  size_t length(ptrdiff_t /*x*/) {
    return 1U;
  }

  namespace agrad {
    // forward declare to avoid header dependencies
    class var;
    template <typename T> class fvar;
  }

  template <typename T>
  size_t length(const stan::agrad::fvar<T>& /*x*/) {
    return 1U;
  }

  size_t length(const stan::agrad::var& /*x*/) {
    return 1U;
  }

}
#endif

