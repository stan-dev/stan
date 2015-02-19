#ifndef STAN__MATH__PRIM__SCAL__META__IS_VAR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_VAR_HPP

#include <stan/math/rev/core/var.hpp>

namespace stan {

  template <typename T>
  struct is_var {
    enum { value = false };
  };
  template <>
  struct is_var<stan::agrad::var> {
    enum { value = true };
  };

}
#endif

