#ifndef STAN_MATH_PRIM_SCAL_META_PARTIALS_RETURN_TYPE_HPP
#define STAN_MATH_PRIM_SCAL_META_PARTIALS_RETURN_TYPE_HPP

#include <stan/math/prim/scal/meta/partials_type.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {

    template <typename T1,
              typename T2 = double,
              typename T3 = double,
              typename T4 = double,
              typename T5 = double,
              typename T6 = double>
    struct partials_return_type {
      typedef typename
      boost::math::tools::promote_args
      <typename partials_type<typename scalar_type<T1>::type>::type,
       typename partials_type<typename scalar_type<T2>::type>::type,
       typename partials_type<typename scalar_type<T3>::type>::type,
       typename partials_type<typename scalar_type<T4>::type>::type,
       typename partials_type<typename scalar_type<T5>::type>::type,
       typename partials_type<typename scalar_type<T6>::type>::type>
      ::type
      type;
    };


}
#endif

