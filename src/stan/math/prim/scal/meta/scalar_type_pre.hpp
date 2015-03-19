#ifndef STAN__MATH__PRIM__SCAL__META__SCALAR_TYPE_PRE_HPP
#define STAN__MATH__PRIM__SCAL__META__SCALAR_TYPE_PRE_HPP

#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/scal/meta/value_type.hpp>

namespace stan {

  namespace {
    template <bool is_vec, typename T, typename T_container>
    struct scalar_type_helper_pre {
      typedef T_container type;
    };

    template <typename T, typename T_container>
    struct scalar_type_helper_pre<true, T, T_container> {
      typedef typename
      scalar_type_helper_pre<is_vector<typename stan::math::value_type<T>::type>::value,
                             typename stan::math::value_type<T>::type,
                             typename stan::math::value_type<T_container>::type>::type
      type;
    };
  }

  /**
    * Metaprogram structure to determine the type of first container of
    * the base scalar type of a template argument.
    *
    * @tparam T Type of object.
  */
  template <typename T>
  struct scalar_type_pre {
    typedef typename
    scalar_type_helper_pre<is_vector<typename stan::math::value_type<T>::type>::value,
                           typename stan::math::value_type<T>::type, T>::type
    type;
  };


}
#endif

