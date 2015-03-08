#ifndef STAN__MATH__FWD__SCAL__META__PARTIALS_TYPE_HPP
#define STAN__MATH__FWD__SCAL__META__PARTIALS_TYPE_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/scal/meta/partials_type.hpp>

namespace stan {

  template <typename T>
  struct partials_type<stan::agrad::fvar<T> > {
    typedef T type;
  };

}
#endif

