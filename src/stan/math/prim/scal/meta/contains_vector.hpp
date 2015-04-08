#ifndef STAN_MATH_PRIM_SCAL_META_CONTAINS_VECTOR_HPP
#define STAN_MATH_PRIM_SCAL_META_CONTAINS_VECTOR_HPP

#include <stan/math/prim/scal/meta/is_vector.hpp>

namespace stan {

  template <typename T1,
            typename T2 = double,
            typename T3 = double,
            typename T4 = double,
            typename T5 = double,
            typename T6 = double>
  struct contains_vector {
    enum {
      value = is_vector<T1>::value
      || is_vector<T2>::value
      || is_vector<T3>::value
      || is_vector<T4>::value
      || is_vector<T5>::value
      || is_vector<T6>::value
    };
  };

}
#endif

