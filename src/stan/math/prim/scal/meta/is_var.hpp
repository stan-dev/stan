#ifndef STAN__MATH__PRIM__SCAL__META__IS_VAR_HPP
#define STAN__MATH__PRIM__SCAL__META__IS_VAR_HPP

namespace stan {

  template <typename T>
  struct is_var {
    enum { value = false };
  };

}
#endif

