#ifndef STAN_INTERFACE_VAR_CONTEXT_TYPEDEFS_HPP
#define STAN_INTERFACE_VAR_CONTEXT_TYPEDEFS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>

namespace stan {
  namespace interface {
    namespace var_context {
      
      typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_t;
      typedef typename stan::math::index_type<vector_t>::type vector_idx_t;
      
      typedef Eigen::Matrix<double, 1, Eigen::Dynamic> row_vector_t;
      typedef typename stan::math::index_type<row_vector_t>::type row_vector_idx_t;
      
      typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
      typedef typename stan::math::index_type<matrix_t>::type matrix_idx_t;
      
      typedef Eigen::Array<int, Eigen::Dynamic, 1> int_array_t;
      typedef typename stan::math::index_type<int_array_t>::type int_array_idx_t;
      
      typedef Eigen::Array<double, Eigen::Dynamic, 1> real_array_t;
      typedef typename stan::math::index_type<real_array_t>::type real_array_idx_t;

    }
  }
}

#endif
