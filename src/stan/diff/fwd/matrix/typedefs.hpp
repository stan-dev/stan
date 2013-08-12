#ifndef __STAN__DIFF__FWD__MATRIX__TYPEDEFS_HPP__
#define __STAN__DIFF__FWD__MATRIX__TYPEDEFS_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/numeric_limits.hpp>

namespace stan {
  namespace diff {
    
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type 
    size_type;

    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic>
    matrix_fv;

    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,1>
    vector_fv;

    typedef 
    Eigen::Matrix<fvar<double>,1,Eigen::Dynamic>
    row_vector_fv;

  }
}
#endif
