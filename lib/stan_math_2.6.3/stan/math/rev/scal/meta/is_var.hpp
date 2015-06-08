#ifndef STAN_MATH_REV_SCAL_META_IS_VAR_HPP
#define STAN_MATH_REV_SCAL_META_IS_VAR_HPP

#include <stan/math/prim/scal/meta/is_var.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {

  template <>
  struct is_var<stan::math::var> {
    enum { value = true };
  };

}
#endif

