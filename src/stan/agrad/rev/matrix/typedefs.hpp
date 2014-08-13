#ifndef STAN__AGRAD__REV__MATRIX__TYPEDEFS_HPP
#define STAN__AGRAD__REV__MATRIX__TYPEDEFS_HPP

#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/Eigen.hpp>

#include "Eigen/src/Core/Matrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "stan/math/matrix/EigenDenseBaseAddons.h"

namespace stan {
  namespace agrad {
class var;

    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;

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
