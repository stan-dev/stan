#ifndef STAN__MATH__REV__SCAL__META__IS_VAR_HPP
#define STAN__MATH__REV__SCAL__META__IS_VAR_HPP


#include <stan/math/prim/scal/meta/is_var.hpp>
#include <stan/math/rev/core/var.hpp>

namespace stan {

  template <>
  struct is_var<stan::agrad::var> {
    enum { value = true };
  };

}
#endif

