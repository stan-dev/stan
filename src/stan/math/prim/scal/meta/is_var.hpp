#ifndef STAN_MATH_PRIM_SCAL_META_IS_VAR_HPP
#define STAN_MATH_PRIM_SCAL_META_IS_VAR_HPP

namespace stan {

  template <typename T>
  struct is_var {
    enum { value = false };
  };

}
#endif

