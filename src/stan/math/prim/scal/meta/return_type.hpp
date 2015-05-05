#ifndef STAN_MATH_PRIM_SCAL_META_RETURN_TYPE_HPP
#define STAN_MATH_PRIM_SCAL_META_RETURN_TYPE_HPP

#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <boost/math/tools/promotion.hpp>

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
  struct return_type {
    typedef typename
    boost::math::tools::promote_args<typename scalar_type<T1>::type,
                                     typename scalar_type<T2>::type,
                                     typename scalar_type<T3>::type,
                                     typename scalar_type<T4>::type,
                                     typename scalar_type<T5>::type,
                                     typename scalar_type<T6>::type>::type
    type;
  };

}
#endif

