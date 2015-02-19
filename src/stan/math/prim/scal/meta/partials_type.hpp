#ifndef STAN__MATH__PRIM__SCAL__META__PARTIALS_TYPE_HPP
#define STAN__MATH__PRIM__SCAL__META__PARTIALS_TYPE_HPP

#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/rev/core/var.hpp>

namespace stan {

  template <typename T>
  struct partials_type {
    typedef T type;
  };
  template <typename T>
  struct partials_type<stan::agrad::fvar<T> > {
    typedef T type;
  };
  template <>
  struct partials_type<stan::agrad::var> {
    typedef double type;
  };

}
#endif

