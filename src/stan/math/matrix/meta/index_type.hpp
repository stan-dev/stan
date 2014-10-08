#ifndef STAN__MATH__MATRIX__META__INDEX_TYPE_HPP
#define STAN__MATH__MATRIX__META__INDEX_TYPE_HPP

#include <stan/math/meta/index_type.hpp>
#include <Eigen/Core>

namespace stan {

  namespace math {

    /**
     * Template metaprogram defining typedef for the type of index for
     * an Eigen matrix, vector, or row vector.
     *
     * @tparam T type of matrix.
     * @tparam R number of rows for matrix.
     * @tparam C number of columns for matrix.
     */
    template <typename T, int R, int C>
    struct index_type<Eigen::Matrix<T, R, C> > {
      typedef typename Eigen::Matrix<T,R,C>::Index type;
    };

    
  }

}


#endif
