#ifndef STAN__MATH__REV__SCAL__META__PARTIALS_TYPE_HPP
#define STAN__MATH__REV__SCAL__META__PARTIALS_TYPE_HPP

#include <stan/math/rev/core/var.hpp>
#include <stan/math/prim/scal/meta/partials_type.hpp>

namespace stan {

  template <>
  struct partials_type<stan::agrad::var> {
    typedef double type;
  };

}
#endif

