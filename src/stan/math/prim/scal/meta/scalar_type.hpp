#ifndef STAN_MATH_PRIM_SCAL_META_SCALAR_TYPE_HPP
#define STAN_MATH_PRIM_SCAL_META_SCALAR_TYPE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/arr/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/value_type.hpp>
#include <stan/math/prim/scal/meta/value_type.hpp>

namespace stan {

  namespace {
    template <bool is_vec, typename T>
    struct scalar_type_helper {
      typedef T type;
    };

    template <typename T>
    struct scalar_type_helper<true, T> {
      typedef typename
      scalar_type_helper<is_vector<typename
                                   stan::math::value_type<T>::type>::value,
                         typename stan::math::value_type<T>::type>::type
      type;
    };
  }
  /**
   * Metaprogram structure to determine the base scalar type
   * of a template argument.
   *
   * <p>This base class should be specialized for structured types.
   *
   * @tparam T Type of object.
   */
  template <typename T>
  struct scalar_type {
    typedef typename scalar_type_helper<is_vector<T>::value, T>::type type;
  };

  // ****************** additions for new VV *************************
  template <typename T>
  struct scalar_type<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > {
    typedef typename scalar_type<T>::type type;
  };

  template <typename T>
  struct scalar_type<T*> {
    typedef typename scalar_type<T>::type type;
  };


}
#endif

