#ifndef STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/rev/core/var.hpp>

namespace stan {

  template <typename T>
  struct is_fvar {
    enum { value = false };
  };
  template <typename T>
  struct is_fvar<stan::agrad::fvar<T> > {
    enum { value = true };
  };

}
#endif

