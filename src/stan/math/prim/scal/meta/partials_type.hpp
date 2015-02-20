#ifndef STAN__MATH__PRIM__SCAL__META__PARTIALS_TYPE_HPP
#define STAN__MATH__PRIM__SCAL__META__PARTIALS_TYPE_HPP

namespace stan {

  template <typename T>
  struct partials_type {
    typedef T type;
  };

}
#endif

