#ifndef STAN_MATH_PRIM_SCAL_META_IS_VAR_OR_ARITHMETIC_HPP
#define STAN_MATH_PRIM_SCAL_META_IS_VAR_OR_ARITHMETIC_HPP

#include <stan/math/prim/scal/meta/is_var.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

namespace stan {

  template <typename T1,
            typename T2 = double,
            typename T3 = double,
            typename T4 = double,
            typename T5 = double,
            typename T6 = double>
  struct is_var_or_arithmetic {
    enum {
      value
      = (is_var<typename scalar_type<T1>::type>::value
         || boost::is_arithmetic<typename scalar_type<T1>::type>::value)
      && (is_var<typename scalar_type<T2>::type>::value
          || boost::is_arithmetic<typename scalar_type<T2>::type>::value)
      && (is_var<typename scalar_type<T3>::type>::value
          || boost::is_arithmetic<typename scalar_type<T3>::type>::value)
      && (is_var<typename scalar_type<T4>::type>::value
          || boost::is_arithmetic<typename scalar_type<T4>::type>::value)
      && (is_var<typename scalar_type<T5>::type>::value
          || boost::is_arithmetic<typename scalar_type<T5>::type>::value)
      && (is_var<typename scalar_type<T6>::type>::value
          || boost::is_arithmetic<typename scalar_type<T6>::type>::value)
    };
  };

}
#endif

