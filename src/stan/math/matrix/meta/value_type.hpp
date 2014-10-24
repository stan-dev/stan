#ifndef STAN__MATH__MATRIX__META__VALUE_TYPE_HPP
#define STAN__MATH__MATRIX__META__VALUE_TYPE_HPP

#include <stan/math/meta/value_type.hpp>
#include <Eigen/Core>

namespace stan {

  namespace math {

    /**
     * Template metaprogram defining the type of values stored in an
     * Eigen matrix, vector, or row vector.
     *
     * @tparam T type of matrix.
     * @tparam R number of rows for matrix.
     * @tparam C number of columns for matrix.
     */
    template <typename T, int R, int C>
    struct value_type<Eigen::Matrix<T, R, C> > {
      typedef typename Eigen::Matrix<T,R,C>::Scalar type;
    };

    
  }

}


#endif
