#ifndef STAN__MATH__REV__MAT__FUN__TYPEDEFS_HPP
#define STAN__MATH__REV__MAT__FUN__TYPEDEFS_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/arr/meta/var.hpp>

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
