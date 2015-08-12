#ifndef STAN_MATH_PRIM_SCAL_META_CONTAINS_FVAR_HPP
#define STAN_MATH_PRIM_SCAL_META_CONTAINS_FVAR_HPP

#include <stan/math/prim/scal/meta/is_fvar.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>

namespace stan {

  /**
   * Metaprogram to calculate the base scalar return type resulting
   * from promoting all the scalar types of the template parameters.
   */
    template <typename T1,
              typename T2 = double,
              typename T3 = double,
              typename T4 = double,
              typename T5 = double,
              typename T6 = double>
    struct contains_fvar {
      enum {
        value = is_fvar<typename scalar_type<T1>::type>::value
        || is_fvar<typename scalar_type<T2>::type>::value
        || is_fvar<typename scalar_type<T3>::type>::value
        || is_fvar<typename scalar_type<T4>::type>::value
        || is_fvar<typename scalar_type<T5>::type>::value
        || is_fvar<typename scalar_type<T6>::type>::value
      };
    };

}
#endif

