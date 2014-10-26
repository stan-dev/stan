#ifndef STAN__AGRAD__REV__MATRIX__TYPEDEFS_HPP
#define STAN__AGRAD__REV__MATRIX__TYPEDEFS_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {

  namespace agrad {

    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Index 
    size_type;

    /**
     * The type of a matrix holding <code>stan::agrad::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>
    matrix_v;

    /**
     * The type of a (column) vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,Eigen::Dynamic,1>
    vector_v;

    /**
     * The type of a row vector holding <code>stan::agrad::var</code>
     * values.
     */
    typedef 
    Eigen::Matrix<var,1,Eigen::Dynamic>
    row_vector_v;

  }
}
#endif
