#ifndef STAN_MATH_PRIM_SCAL_META_CONTAINS_NONCONSTANT_STRUCT_HPP
#define STAN_MATH_PRIM_SCAL_META_CONTAINS_NONCONSTANT_STRUCT_HPP

#include <stan/math/prim/scal/meta/is_constant_struct.hpp>

namespace stan {

  template <typename T1,
            typename T2 = double,
            typename T3 = double,
            typename T4 = double,
            typename T5 = double,
            typename T6 = double>
  struct contains_nonconstant_struct {
    enum {
      value = !is_constant_struct<T1>::value
      || !is_constant_struct<T2>::value
      || !is_constant_struct<T3>::value
      || !is_constant_struct<T4>::value
      || !is_constant_struct<T5>::value
      || !is_constant_struct<T6>::value
    };
  };

}
#endif

