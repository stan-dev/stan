#ifndef STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_FVAR_HPP

namespace stan {

  template <typename T>
  struct is_fvar {
    enum { value = false };
  };

}
#endif

