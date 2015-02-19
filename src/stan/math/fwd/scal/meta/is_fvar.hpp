#ifndef STAN__MATH__FWD__SCAL__META__IS_FVAR_HPP
#define STAN__MATH__FWD__SCAL__META__IS_FVAR_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/prim/scal/meta/is_fvar.hpp>

namespace stan {

  template <typename T>
  struct is_fvar<stan::agrad::fvar<T> > {
    enum { value = true };
  };

}
#endif
