#ifndef STAN__AGRAD__FWD__MATRIX__TYPEDEFS_HPP
#define STAN__AGRAD__FWD__MATRIX__TYPEDEFS_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/fwd/numeric_limits.hpp>

namespace stan {
  namespace agrad {
    
    typedef 
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::Index 
    size_type;

    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,Eigen::Dynamic>
    matrix_fd;

    typedef 
    Eigen::Matrix<fvar<var>,Eigen::Dynamic,Eigen::Dynamic>
    matrix_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,Eigen::Dynamic>
    matrix_ffd;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,Eigen::Dynamic>
    matrix_ffv;

    typedef 
    Eigen::Matrix<fvar<double>,Eigen::Dynamic,1>
    vector_fd;

    typedef 
    Eigen::Matrix<fvar<var>,Eigen::Dynamic,1>
    vector_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<double> >,Eigen::Dynamic,1>
    vector_ffd;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,Eigen::Dynamic,1>
    vector_ffv;

    typedef 
    Eigen::Matrix<fvar<double>,1,Eigen::Dynamic>
    row_vector_fd;

    typedef 
    Eigen::Matrix<fvar<var>,1,Eigen::Dynamic>
    row_vector_fv;

    typedef 
    Eigen::Matrix<fvar<fvar<double> >,1,Eigen::Dynamic>
    row_vector_ffd;

    typedef 
    Eigen::Matrix<fvar<fvar<var> >,1,Eigen::Dynamic>
    row_vector_ffv;

  }
}
#endif
