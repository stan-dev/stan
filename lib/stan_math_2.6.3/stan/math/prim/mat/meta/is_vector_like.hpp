#ifndef STAN_MATH_PRIM_MAT_META_IS_VECTOR_LIKE_HPP
#define STAN_MATH_PRIM_MAT_META_IS_VECTOR_LIKE_HPP

#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  // handles eigen matrix
  template <typename T>
  struct is_vector_like<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > {
    enum { value = true };
  };
}
#endif

