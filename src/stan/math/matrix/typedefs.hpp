#ifndef STAN__MATH__MATRIX__TYPEDEFS_HPP
#define STAN__MATH__MATRIX__TYPEDEFS_HPP

#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    // typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type size_type;
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Index size_type;

    /**
     * Type for matrix of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>
    matrix_d;

    /**
     * Type for (column) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,1>
    vector_d;

    /**
     * Type for (row) vector of double values.
     */
    typedef 
    Eigen::Matrix<double,1,Eigen::Dynamic>
    row_vector_d;

  }
}

#endif
