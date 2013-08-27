#ifndef __STAN__DIFF__REV__MATRIX__TYPEDEFS_HPP__
#define __STAN__DIFF__REV__MATRIX__TYPEDEFS_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/rev/var.hpp>

namespace stan {
  namespace diff {
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;

    /**
     * The type of a matrix holding <code>stan::diff::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>
    matrix_v;

    /**
     * The type of a (column) vector holding <code>stan::diff::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,1>
    vector_v;

    /**
     * The type of a row vector holding <code>stan::diff::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,1,Eigen::Dynamic>
    row_vector_v;

  }
}
#endif
