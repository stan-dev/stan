#ifndef STAN__MATH__PRIM__MAT__META__IS_VECTOR_LIKE_HPP
#define STAN__MATH__PRIM__MAT__META__IS_VECTOR_LIKE_HPP

#include <stan/math/prim/scal/meta/is_vector_like.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {

  // handles eigen matrix
  template <typename T>
  struct is_vector_like<Eigen::MatrixBase<T> > {
    enum { value = true };
  };
  template <typename T, int R, int C>
  struct is_vector_like<Eigen::Matrix<T,R,C> > {
    enum { value = true };
  };
}
#endif

