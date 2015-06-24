#ifndef STAN_MATH_PRIM_SCAL_META_IS_FVAR_HPP
#define STAN_MATH_PRIM_SCAL_META_IS_FVAR_HPP

namespace stan {

  template <typename T>
  struct is_fvar {
    enum { value = false };
  };

}
#endif

