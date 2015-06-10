#ifndef STAN_MATH_PRIM_SCAL_META_IS_CONSTANT_STRUCT_HPP
#define STAN_MATH_PRIM_SCAL_META_IS_CONSTANT_STRUCT_HPP

#include <stan/math/prim/scal/meta/is_constant.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <vector>


namespace stan {

  /**
   * Metaprogram to determine if a type has a base scalar
   * type that can be assigned to type double.
   */
  template <typename T>
  struct is_constant_struct {
    enum { value = is_constant<T>::value };
  };

  template <typename T>
  struct is_constant_struct<std::vector<T> > {
    enum { value = is_constant_struct<T>::value };
  };

  template <typename T, int R, int C>
  struct is_constant_struct<Eigen::Matrix<T, R, C> > {
    enum { value = is_constant_struct<T>::value };
  };

  template <typename T>
  struct is_constant_struct<Eigen::Block<T> > {
    enum { value = is_constant_struct<T>::value };
  };

}
#endif

