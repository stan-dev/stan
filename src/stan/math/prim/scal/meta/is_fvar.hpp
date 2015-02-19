#ifndef STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP

#include <stan/math/fwd/core/fvar.hpp>

namespace stan {

  template <typename T>
  struct is_fvar {
    enum { value = false };
  };

}
#endif

